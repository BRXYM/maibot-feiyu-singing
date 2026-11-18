"""Preserve measured notes and align ASR syllable windows, including melismas."""
from pathlib import Path
import json
import math
import re

def repair_character_timing(original, text):
    """Map corrected syllables monotonically onto measured ASR timing anchors."""
    import difflib
    from bisect import bisect_right
    from pypinyin import lazy_pinyin
    if not text or abs(len(text)-len(original))>max(3,int(len(original)*.25)):
        raise ValueError('歌词纠错改动过大，无法可靠对齐')
    if len(text)==len(original):
        return [dict(c,asr_text=c['text'],text=t) for c,t in zip(original,text)]
    before=lazy_pinyin(''.join(c['text'] for c in original))
    after=lazy_pinyin(text)
    matcher=difflib.SequenceMatcher(None,before,after,autojunk=False)
    anchors={-.5:original[0]['start'],len(text)-.5:original[-1]['end']}
    matches=0
    for a,b,size in matcher.get_matching_blocks():
        for j in range(size):
            anchors[b+j]=(original[a+j]['start']+original[a+j]['end'])/2
            matches+=1
    if matches<min(len(original),len(text))*.45:
        raise ValueError('歌词纠错与实测发音不够相近')
    xs=sorted(anchors)
    centers=[]
    for i in range(len(text)):
        j=max(0,min(len(xs)-2,bisect_right(xs,i)-1))
        a,b=xs[j:j+2]
        centers.append(anchors[a]+(anchors[b]-anchors[a])*(i-a)/(b-a))
    bounds=[original[0]['start']]+[(a+b)/2 for a,b in zip(centers,centers[1:])]+[original[-1]['end']]
    if any(b-a<.045 for a,b in zip(bounds,bounds[1:])):
        raise ValueError('修正后的唱字时间过短')
    return [{'text':c,'start':bounds[i],'end':bounds[i+1],'timing':'interpolated between matching observed pinyin anchors'} for i,c in enumerate(text)]

HAN=re.compile(r'[\u3400-\u9fff]')

def measured_phone_durations(ds,events,phones):
    """Keep measured consonant/vowel boundaries inside each measured sung word."""
    groups=[]
    for event in events:
        if not event['slur']:
            groups.append(dict(event))
        else:
            groups[-1]['end']=event['end']
    durations=[]
    for group in groups:
        a,b=group['start'],group['end']
        if group['text']=='SP':
            durations.append(b-a)
            continue
        matched=[p for p in phones if p['text'] not in ['SP','AP'] and p['start']>=a-.002 and p['end']<=b+.002]
        if len(matched)!=len(group['phonemes']):
            raise ValueError('原唱音素与声库音素数量不一致，不能可靠保留咬字时长')
        bounds=[a]+[(x['end']+y['start'])/2 for x,y in zip(matched,matched[1:])]+[b]
        values=[y-x for x,y in zip(bounds,bounds[1:])]
        if any(x<=0 or not math.isfinite(x) for x in values):
            raise ValueError('原唱音素时长无效')
        durations.extend(values)
    if len(durations)!=len(ds['ph_seq'].split()) or abs(sum(durations)-sum(map(float,ds['note_dur'].split())))>.002:
        raise ValueError('原唱音素总时长与乐谱不一致')
    ds['ph_dur']=' '.join(f'{d:.6f}' for d in durations)
    return ds

def chars_from_words(words, duration):
    chars=[]
    for word in words:
        text=''.join(HAN.findall(word.get('word','')))
        start=max(0.0,float(word['start']))
        end=min(duration,float(word['end']))
        if not text or end<=start:
            continue
        for i,char in enumerate(text):
            chars.append({'text':char,'start':start+(end-start)*i/len(text),'end':start+(end-start)*(i+1)/len(text)})
    if not chars:
        raise ValueError('没有可对齐的中文歌词')
    for a,b in zip(chars,chars[1:]):
        if b['start'] < a['end']-1e-4:
            raise ValueError('ASR 歌词时间相互重叠')
    return chars

def build_score(notes, chars, duration, dictionary):
    """One word per Han character; several notes may share one sustained vowel."""
    from pypinyin import lazy_pinyin, Style
    py=lazy_pinyin(''.join(c['text'] for c in chars),style=Style.NORMAL)
    events=[]
    cursor=0.0
    for char,pinyin in zip(chars,py):
        a,b=char['start'],char['end']
        if b-a<.045:
            raise ValueError('唱字短于45ms，拒绝不可靠的歌词对齐')
        if a-cursor>.02:
            events.append({'text':'SP','phonemes':['SP'],'start':cursor,'end':a,'pitch':None,'slur':0})
        elif events:
            events[-1]['end']=a
        if pinyin not in dictionary:
            raise ValueError(f'声库字典缺少读音：{pinyin}')
        matches=[n for n in notes if min(b,n['end'])-max(a,n['start'])>.015]
        if not matches:
            distance=lambda n: min(abs(a-n['end']),abs(b-n['start']))
            nearest=min(notes,key=distance)
            if distance(nearest)>.3:
                raise ValueError('歌词附近没有识别到旋律，无法可靠合成')
            matches=[nearest]
        # Measured pitch transitions remain; tiny ornaments are merged into a longer note.
        points=[a]+[max(a,min(b,(x['end']+y['start'])/2)) for x,y in zip(matches,matches[1:])]+[b]
        pieces=[]
        for i,n in enumerate(matches):
            start,end=points[i],points[i+1]
            if end-start<.04:
                continue
            pitch=int(round(n['pitch']))
            if not 24<=pitch<=96:
                raise ValueError('提取音高超出合理演唱范围')
            if pieces and pieces[-1]['pitch']==pitch:
                pieces[-1]['end']=end
            else:
                pieces.append({'text':char['text'],'phonemes':dictionary[pinyin].split(),'start':start,'end':end,'pitch':pitch,'slur':int(bool(pieces))})
        if not pieces:
            raise ValueError('唱字内的音符过碎，需要重新选择片段')
        pieces[0]['start']=a
        pieces[-1]['end']=b
        events+=pieces
        cursor=b
    if duration-cursor>.02:
        events.append({'text':'SP','phonemes':['SP'],'start':cursor,'end':duration,'pitch':None,'slur':0})
    def note_name(p):
        if p is None:
            return 'rest'
        return ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][p%12]+str(p//12-1)
    words=[e for e in events if not e['slur']]
    ds={'offset':0,'text':' '.join(e['text'] for e in words),'ph_seq':' '.join(p for e in words for p in e['phonemes']),
        'ph_num':' '.join(str(len(e['phonemes'])) for e in words),'note_seq':' '.join(note_name(e['pitch']) for e in events),
        'note_slur':' '.join(str(e['slur']) for e in events),'note_dur':' '.join(f"{e['end']-e['start']:.6f}" for e in events),'lang':'zh'}
    return ds,events
