"""Validate contiguous, word-safe chunks before rendering any audio."""
import math


def full_song_parts(transcript, duration):
    """Group sung phrases into bounded windows, leaving long instrumental gaps open."""
    words = sorted((float(w['start']), float(w['end']))
                   for s in transcript['segments'] for w in s.get('words', [])
                   if 0 <= float(w['start']) < float(w['end']) <= duration + .1)
    if not words:
        raise ValueError('原曲没有可定位的唱句')
    phrases = []
    start, end = words[0]
    for word_start, word_end in words[1:]:
        if word_start - end >= 4:
            phrases.append((start, end))
            start = word_start
        end = max(end, word_end)
    phrases.append((start, end))
    phrase_ends = [float(segment['end']) for segment in transcript['segments'] if segment.get('words')]
    parts = []
    for start, end in phrases:
        cursor = max(0, start - .2)
        end = min(duration, end + .2)
        while end - cursor > 30:
            cuts = [(a + b) / 2 for (_, a), (b, _) in zip(words, words[1:])
                    if a <= b and cursor + 8 <= (a + b) / 2 <= cursor + 29.8]
            if not cuts:
                raise ValueError('未能在30秒内找到不切断唱字的分界')
            preferred = [cut for cut in cuts if cut >= cursor + 21
                         and end - cut >= 5
                         and any(abs(cut - boundary) <= .45 for boundary in phrase_ends)]
            point = max(preferred or cuts)
            parts.append({'start': cursor, 'end': point, 'section': '整首'})
            cursor = point
        if end > cursor:
            parts.append({'start': cursor, 'end': end, 'section': '整首'})
    return parts

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
