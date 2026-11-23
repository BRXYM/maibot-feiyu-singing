"""Review brief pitch excursions against periodicity in the isolated vocal."""
import numpy as np
from scipy.signal import correlate, find_peaks


def periodic_pitch(audio, center, rate):
    # A local normalized autocorrelation provides a second, independent estimate.
    radius = round(.024 * rate)
    wave = np.asarray(audio[max(0, center-radius):center+radius], dtype=np.float64)
    if len(wave) < 2*radius or np.max(np.abs(wave)) < .001:
        return 0.
    wave = wave - wave.mean()
    lo, hi = max(2, rate//1200), min(rate//55, len(wave)//2)
    correlation = correlate(wave, wave, mode='full', method='fft')[len(wave)-1:len(wave)+hi]
    energy = np.r_[0., np.cumsum(wave**2)]
    lags = np.arange(hi+1)
    score = correlation / np.sqrt(np.maximum(energy[len(wave)-lags]*(energy[-1]-energy[lags]), 1e-20))
    peaks = find_peaks(score)[0]
    peaks = peaks[peaks >= lo]
    if not peaks.size:
        return 0.
    strong = peaks[score[peaks] >= max(.78, .94 * score[peaks].max())]
    return rate / strong[0] if strong.size else 0.


def review_pitch(audio, f0, rate, hop):
    contour = np.asarray(f0)
    output = contour.copy()
    notes = np.full(len(contour), np.nan)
    voiced = contour > 0
    notes[voiced] = 12*np.log2(contour[voiced])
    starts = np.flatnonzero(np.abs(np.diff(notes)) >= 3.5) + 1
    max_frames = round(.08*rate/hop)
    checked = repaired = frames = 0
    last = 0
    for start in starts:
        if start < max(3,last) or start+3 >= len(notes):
            continue
        before = notes[start-3:start]
        if not np.isfinite(before).all() or np.ptp(before) > 1.2:
            continue
        base = before[-1]
        for end in range(start+1,min(start+max_frames+1,len(notes)-2)):
            if not np.isfinite(notes[end-1:end+1]).all():
                break
            if abs(notes[end]-base) > 1.2:
                continue
            after = notes[end:end+3]
            if (not np.isfinite(after).all() or np.ptp(after)>1.2 or
                    np.min(np.abs(notes[start:end]-base)) < 2.5):
                break
            checked += 1
            centers = np.arange(start,end)
            guide = np.interp(centers,[start-1,end],[base,notes[end]])
            votes = []
            confirms_excursion = False
            for idx, expected in zip(centers, guide):
                pitch = periodic_pitch(audio,int(idx)*hop,rate)
                if pitch and abs(12*np.log2(pitch)-notes[idx]) <= 1.2:
                    confirms_excursion = True
                votes.append(bool(pitch and abs(12*np.log2(pitch)-expected) <= 1.2
                                  and abs(12*np.log2(pitch)-notes[idx]) >= 2.5))
            if not confirms_excursion and sum(votes) >= np.ceil(.75*len(votes)):
                output[start:end] = np.exp2(guide/12)
                repaired += 1
                frames += int(end-start)
            last = end
            break
    return output, dict(candidates=checked, corrected=repaired, corrected_frames=frames)
