"""Public timed-lyric metadata; timing of the selected audio still comes from ASR."""
import difflib
import html
import re
import httpx
from pypinyin import lazy_pinyin

HAN=re.compile(r'[\u3400-\u9fff]')

def parse_lrc(raw):
    lines=[]
    for line in html.unescape(raw).splitlines():
        timestamps=re.findall(r'\[(\d+):(\d+(?:\.\d+)?)\]',line)
        text=re.sub(r'\[[^]]*\]','',line).strip()
        if not timestamps or not text or any(k in text for k in ['作词','作曲','编曲','制作人','录音','混音']):
            continue
        for minute,second in timestamps:
            lines.append({'time':int(minute)*60+float(second),'text':text})
    if len(lines)<5:
        raise ValueError('公开歌词不足以支持可靠对齐')
    return sorted(lines,key=lambda r:r['time'])

def netease_lyrics(song,artist):
    with httpx.Client(headers={'Referer':'https://music.163.com/','User-Agent':'Mozilla/5.0'},timeout=25,trust_env=False) as client:
        response=client.get('https://music.163.com/api/search/get',params={'s':song+' '+artist,'type':1,'limit':10,'offset':0})
        response.raise_for_status()
        entries=response.json().get('result',{}).get('songs',[])
        candidates=[s for s in entries if s['name'].strip()==song.strip() and (not artist or any(artist.strip()==a['name'].strip() for a in s['artists']))]
        if not candidates:
            raise ValueError('没有找到与歌名、原唱一致的公开歌词元数据')
        entry=candidates[0]
        response=client.get('https://music.163.com/api/song/lyric',params={'id':entry['id'],'lv':1,'kv':1,'tv':-1})
        response.raise_for_status()
        raw=response.json().get('lrc',{}).get('lyric','')
        return {'song_id':entry['id'],'song':entry['name'],'artist':[a['name'] for a in entry['artists']],
                'source_url':'https://music.163.com/song?id='+str(entry['id']),'provider':'netease','lines':parse_lrc(raw)}

def qq_lyrics(song,artist):
    with httpx.Client(headers={'Referer':'https://y.qq.com/','User-Agent':'Mozilla/5.0'},timeout=25,trust_env=False) as client:
        response=client.get('https://c.y.qq.com/soso/fcgi-bin/client_search_cp',params={'w':song+' '+artist,'format':'json','p':1,'n':20})
        response.raise_for_status()
        entries=response.json().get('data',{}).get('song',{}).get('list',[])
        candidates=[s for s in entries if s['songname'].strip()==song.strip() and (not artist or any(artist.strip()==a['name'].strip() for a in s.get('singer',[])))]
        if not candidates:
            raise ValueError('QQ音乐没有匹配歌名及原唱的歌词曲目')
        entry=candidates[0]
        response=client.get('https://c.y.qq.com/lyric/fcgi-bin/fcg_query_lyric_new.fcg',params={'songmid':entry['songmid'],'format':'json','nobase64':1})
        response.raise_for_status()
        return {'song_id':entry['songmid'],'song':entry['songname'],'artist':[a['name'] for a in entry['singer']],
                'provider':'qq_music','source_url':'https://y.qq.com/n/ryqq/songDetail/'+entry['songmid'],'lines':parse_lrc(response.json().get('lyric',''))}

def fetch_lyrics(song,artist):
    failures=[]
    for provider in [netease_lyrics,qq_lyrics]:
        try:
            result=provider(song,artist)
            result['source_attempts']=failures
            return result
        except (httpx.HTTPError,ValueError,KeyError,TypeError) as exc:
            failures.append({'provider':provider.__name__,'error':str(exc)[:300]})
    raise ValueError('公开歌词来源均未能取得匹配资料：'+repr(failures))

def candidate_spans(observed,lines,limit=8,min_lines=1):
    # Match pinyin, so homophone mistakes and traditional/simplified ASR do not
    # require inventing new lyrics. Keep whole verified source substrings.
    canonical=''
    boundaries=[0]
    for row in lines:
        text=''.join(HAN.findall(row['text']))
        if text:
            canonical+=text
            boundaries.append(len(canonical))
    syllables=lazy_pinyin(canonical)
    needle=lazy_pinyin(observed)
    ranked={}
    radius=max(3,int(len(needle)*.22))
    for i,start in enumerate(boundaries[:-1]):
        for j,end in enumerate(boundaries):
            if j-i<min_lines:
                continue
            length=end-start
            if not max(4,len(needle)-radius)<=length<=len(needle)+radius:
                continue
            score=difflib.SequenceMatcher(None,needle,syllables[start:end],autojunk=False).ratio()
            score-=.03*abs(length-len(needle))/max(1,len(needle))
            text=canonical[start:end]
            if text not in ranked or score>ranked[text]['score']:
                ranked[text]={'text':text,'score':score,'start_char':start,'end_char':end,'line_count':j-i}
    result=sorted(ranked.values(),key=lambda r:r['score'],reverse=True)[:limit]
    if not result or result[0]['score']<.45:
        raise ValueError('识别唱句与公开歌词无法对应')
    return result
