"""Validate contiguous, word-safe chunks before rendering any audio."""
import math

def validate_parts(parts,selection,transcript,maximum=30):
    if not isinstance(parts,list) or not 2<=len(parts)<=4:
        raise ValueError('长段学唱须分为2–4段')
    result=[]
    cursor=float(selection['start'])
    words=[w for s in transcript['segments'] for w in s.get('words',[])]
    for i,part in enumerate(parts):
        start,end=float(part['start']),float(part['end'])
        if not all(math.isfinite(x) for x in [start,end]) or abs(start-cursor)>.025 or not 5<=end-start<=maximum+.001:
            raise ValueError('分段时间不连续或超出5–30秒范围')
        if i<len(parts)-1 and any(float(w['start'])+.03<end<float(w['end'])-.03 for w in words):
            raise ValueError('分段边界切进了唱字')
        result.append({'start':cursor,'end':end,'section':selection.get('section','副歌'),'reason':'AI按完整唱句分段'})
        cursor=end
    if abs(cursor-float(selection['end']))>.025:
        raise ValueError('分段未覆盖完整选段')
    return result
