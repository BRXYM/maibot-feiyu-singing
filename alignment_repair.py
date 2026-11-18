"""Resolve collapsed syllables using independently observed, exact ASR anchors."""
from copy import deepcopy
from difflib import SequenceMatcher

def repair_collapsed_syllables(alignment,observed):
    original=alignment['chars']
    collapsed=[i for i,c in enumerate(original) if c['end']-c['start']<.045]
    if not collapsed:
        return alignment,[]
    if len(collapsed)>max(1,len(original)//20):
        raise ValueError('多个唱字对齐塌缩，不能可靠修复')
    anchors={}
    matcher=SequenceMatcher(None,''.join(c['text'] for c in original),''.join(c['text'] for c in observed),autojunk=False)
    for block in matcher.get_matching_blocks():
        if block.size>=3:
            anchors.update({block.a+j:observed[block.b+j] for j in range(block.size)})
    repaired=deepcopy(alignment)
    chars=repaired['chars']
    changes=[]
    for i in collapsed:
        if i==0 or i==len(chars)-1 or not all(j in anchors for j in [i-1,i,i+1]):
            raise ValueError('塌缩唱字缺少连续正确歌词的实测时间锚点')
        # A collapsed word commonly means a boundary drift spanning its two
        # neighbours. Replace the whole measured local phrase, not just one word.
        for j in [i-1,i,i+1]:
            a,b=anchors[j]['start'],anchors[j]['end']
            if not .06<=b-a<=1.2 or abs((a+b)-(original[j]['start']+original[j]['end']))/2>.75:
                raise ValueError('实测唱字锚点与歌声对齐不一致')
            chars[j].update(start=a,end=b)
            changes.append({'index':j,'text':chars[j]['text'],'before':[original[j]['start'],original[j]['end']],'after':[a,b],'evidence':'exact >=3-character ASR phrase with measured timestamps'})
        if i>=2 and chars[i-2]['end']>chars[i-1]['start']:
            chars[i-2]['end']=chars[i-1]['start']
        if i+2<len(chars) and chars[i+2]['start']<chars[i+1]['end']:
            chars[i+2]['start']=chars[i+1]['end']
    if any(c['end']-c['start']<.045 for c in chars) or any(a['end']>b['start']+.00001 for a,b in zip(chars,chars[1:])):
        raise ValueError('局部时间修正后仍有塌缩或交叠，拒绝生成')
    # Rescale phones only inside the changed measured word windows. Preserve
    # consonant/vowel order, gaps, sentence length, and every unchanged word.
    for before,after in zip(original,chars):
        if before==after:
            continue
        width=before['end']-before['start']
        if width<=0:
            raise ValueError('原始音素时间无效')
        for old_phone,phone in zip(alignment['phones'],repaired['phones']):
            if before['start']-.000001<=old_phone['start']<old_phone['end']<=before['end']+.000001:
                phone['start']=after['start']+(old_phone['start']-before['start'])/width*(after['end']-after['start'])
                phone['end']=after['start']+(old_phone['end']-before['start'])/width*(after['end']-after['start'])
    return repaired,changes
