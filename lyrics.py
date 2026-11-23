"""Public timed-lyric metadata; timing of the selected audio still comes from ASR."""
from pathlib import Path
import difflib
import html
import json
import re
import unicodedata
import httpx
from pypinyin import lazy_pinyin

HAN=re.compile(r'[\u3400-\u9fff]')

CREDIT = re.compile(
    r'^(?:(?:作[词詞曲]|[词詞曲]|[编編]曲|制作人|[录錄]音(?:室|师|師)?|混音(?:师|師)?|监制|'
    r'歌曲(?:介绍|简介)|专辑(?:介绍|简介)|歌手|演唱|发行(?:公司)?|出品(?:公司)?|版权|文案|旁白|独白|念白|对白|'
    r'recording engineers?|mixing engineers?|(?:recording|mixing) studios?|musicians?|guitars?|bass|drums?|'
    r'background vocals?(?: arrangement)?|producer|lyrics(?: by)?|composed by|spoken(?: intro| outro)?|narration|op|sp)\s*[:：]|'
    r'ISRC\s*[:： ]\s*[A-Z]{2}[- ]?[A-Z0-9]{3})', re.I)
SPOKEN = re.compile(r'^[\[【(（](?:旁白|独白|念白|对白|歌曲介绍|spoken(?: intro| outro| word)?|narration)[\]】)）]', re.I)
SECTION = re.compile(r'^[\[【](?:verse|chorus|bridge|intro|outro|pre-chorus|rap|主歌|副歌|桥段)(?:\s*\d+)?[\]】]\s*', re.I)
RELEASE_NOTE = re.compile(r'^(?:本首歌(?:曲)?|本曲|这首歌曲|该曲|歌曲《[^》]{1,80}》)(?:收录[于在]|发行于|由.{1,24}(?:作词|作曲|制作|演唱))')


def clean_line(value):
    value = value.strip()
    if CREDIT.match(value) or RELEASE_NOTE.match(value) or SPOKEN.match(value) or re.fullmatch(r'\[(?:ar|ti|al|by|offset|length|re|ve):[^\]]*\]',value,re.I):
        return ''
    value = SECTION.sub('', value).strip()
    return '' if re.fullmatch(r'[\[【(（][^\]】)）]{1,40}[\]】)）]',value) and any(
        word in value.casefold() for word in ('间奏','前奏','尾奏','instrumental','music break')) else value


def clean_reference(data):
    titles = set()
    song = normalized_name(data.get('song',''))
    for artist in data.get('artist',[]):
        name = normalized_name(artist)
        titles.update(((song,name),(name,song)))
    lines = []
    for row in data.get('lines',[]):
        text = clean_line(row['text'])
        if text and tuple(re.split(r'\s*[-–—]\s*',normalized_name(text),maxsplit=1)) not in titles:
            lines.append({**row,'text':text})
    if len(lines) < 5:
        raise ValueError('清理后可用的演唱歌词不足')
    return {**data, 'lines':lines}


def normalized_name(value: str) -> str:
    return unicodedata.normalize('NFKC', value).strip().casefold()


def cached_verified_lyrics(jobs: Path, song: str, artist: str) -> dict | None:
    """Reuse source-backed timed lyrics only from earlier successful jobs."""
    candidates = []
    identities = set()
    for path in jobs.glob('*/verified-lyrics.json'):
        try:
            if not re.fullmatch('[0-9a-f]{32}', path.parent.name):
                continue
            lyrics = json.loads(path.read_text('utf-8'))
            if normalized_name(lyrics['song']) != normalized_name(song) or not lyrics['source_url']:
                continue
            names = frozenset(normalized_name(name) for name in lyrics['artist'] if isinstance(name, str) and name.strip())
            if not names or (artist and normalized_name(artist) not in names):
                continue
            lines = lyrics['lines']
            if len(lines) < 5 or not all(isinstance(line['time'], (int, float)) and line['time'] >= 0
                                         and isinstance(line['text'], str) and line['text'].strip() for line in lines):
                continue
            result = json.loads(path.with_name('result.json').read_text('utf-8'))
            if result.get('status') != 'success':
                continue
            candidates.append((float(result.get('finished_at') or path.stat().st_mtime), clean_reference(lyrics)))
            identities.add(names)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    if candidates and (artist or len(identities) == 1):
        return max(candidates, key=lambda row: row[0])[1]
    return None


def parse_lrc(raw):
    lines=[]
    # Only suppress an entire marked spoken block when a later sung-section tag
    # gives it a clear end. Some LRC files mark just the opening spoken line.
    entries = html.unescape(raw).splitlines()
    time_tag = re.compile(r'\[\d+:\d+(?:\.\d+)?\]')
    last_sung_marker = max((i for i,line in enumerate(entries) if SECTION.match(time_tag.sub('',line).strip())),default=-1)
    in_spoken_block = False
    for i,line in enumerate(entries):
        content = time_tag.sub('',line).strip()
        if SECTION.match(content):
            in_spoken_block = False
        elif SPOKEN.fullmatch(content) and i < last_sung_marker:
            in_spoken_block = True
        if in_spoken_block:
            continue
        timestamps=re.findall(r'\[(\d+):(\d+(?:\.\d+)?)\]',line)
        text=clean_line(content)
        if not timestamps or not text:
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

def lrclib_lyrics(song, artist):
    def normalized(value):
        return unicodedata.normalize('NFKC', value).strip().casefold()

    with httpx.Client(timeout=12, headers={'User-Agent': 'FeiyuSinging/1.1 (MaiBot lyric lookup)'}) as client:
        response = client.get('https://lrclib.net/api/search', params={'track_name': song, 'artist_name': artist})
        response.raise_for_status()
        for row in response.json():
            if (normalized(row.get('trackName', '')) != normalized(song)
                    or (artist and normalized(row.get('artistName', '')) != normalized(artist))
                    or row.get('instrumental') or not row.get('syncedLyrics')):
                continue
            try:
                lines = parse_lrc(row['syncedLyrics'])
            except ValueError:
                continue
            return {'song_id': row['id'], 'song': row['trackName'], 'artist': [row['artistName']],
                    'source_url': 'https://lrclib.net/api/get/' + str(row['id']), 'provider': 'lrclib', 'lines': lines}
    raise ValueError('LRCLIB 未找到同名同歌手的时间歌词')


PROVIDER_KEYS = {'netease_lyrics': 'netease', 'qq_lyrics': 'qq_music', 'lrclib_lyrics': 'lrclib'}


def fetch_lyrics(song,artist, *, accept=None, cached=None):
    failures=[]
    if cached is not None:
        try:
            cached=clean_reference(cached)
            if accept:
                accept(cached)
            return cached
        except ValueError as exc:
            failures.append({'provider':cached.get('provider','cached'),'error':str(exc)[:300]})
    for provider in (netease_lyrics,qq_lyrics,lrclib_lyrics):
        if cached and cached.get('provider') == PROVIDER_KEYS.get(provider.__name__):
            continue
        try:
            result=clean_reference(provider(song,artist))
            if accept:
                accept(result)
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
