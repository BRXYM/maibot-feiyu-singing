"""Choose one vocal octave per performance; all times and pitches stay relative."""
from math import gcd

import numpy as np
import pyworld
import soundfile as sf
from scipy.signal import resample_poly


# Working tessitura for Feiyu, in MIDI notes. These are soft scoring boundaries.
LOW, HIGH = 55, 77  # G3 through F5
STEP = .02


def choose_octave(notes):
    notes = np.asarray(notes, dtype=float)
    result = {'semitones': 0, 'comfortable_midi': [LOW, HIGH], 'voiced_seconds': round(len(notes)*STEP, 2)}
    if len(notes)*STEP < 1:
        return result
    p10, center, p90 = np.percentile(notes, [10, 50, 90])
    result['source_midi_p10_p50_p90'] = [round(float(v), 1) for v in (p10, center, p90)]
    # Each voiced frame has equal weight. Winsorize the tails so an isolated
    # high note, a growl or an octave-tracking error cannot dominate the cost.
    body = np.clip(notes, p10, p90)
    def cost(shift):
        shifted = body + shift
        outside = np.maximum(LOW-shifted, 0) + np.maximum(shifted-HIGH, 0)
        return float(np.mean(outside)) + .10*abs(center+shift-(LOW+HIGH)/2)
    best = min((0, 12, -12), key=cost)
    if cost(0)-cost(best) >= 1 and np.mean((notes < LOW) | (notes > HIGH)) >= .25:
        result['semitones'] = best
    return result


def analyze_vocals(path, dialogue=(), origin=0):
    pitches, levels = [], []
    with sf.SoundFile(path) as src:
        rate = src.samplerate
        factor = gcd(rate, 16000)
        while src.tell() < len(src):
            start = src.tell()/rate + origin
            audio = src.read(8*rate, dtype='float64', always_2d=True).mean(axis=1)
            if len(audio) < rate*.1 or np.max(np.abs(audio)) < .0001:
                continue
            wave = resample_poly(audio, 16000//factor, rate//factor)
            f0, times = pyworld.dio(wave, 16000, f0_floor=55, f0_ceil=1100, frame_period=STEP*1000)
            f0 = pyworld.stonemask(wave, f0, times, 16000)
            positions = np.rint(times*16000).astype(int)
            energy = np.r_[0., np.cumsum(wave**2)]
            left, right = np.maximum(positions-160,0), np.minimum(positions+160,len(wave))
            rms = np.sqrt(np.maximum(energy[right]-energy[left],0)/np.maximum(right-left,1))
            voiced = (f0 >= 55) & (f0 <= 1100) & (rms > .0001)
            for a,b in dialogue:
                voiced &= (start+times < a) | (start+times >= b)
            levels.extend(rms[voiced])
            pitches.extend(f0[voiced])
    if not levels:
        return choose_octave([])
    loud_enough = np.asarray(levels) >= np.percentile(levels,90)*.05
    return choose_octave(69+12*np.log2(np.asarray(pitches)[loud_enough]/440))
