import json
import numpy as np

from pitch_review import review_pitch


def test_brief_octave_needs_independent_audio_evidence():
    rate, hop = 16000, 160
    time = np.arange(rate)/rate
    estimate = np.full(100,220.)
    estimate[40:45] = 440.
    for sung, expected_count in ((np.full(rate,220.),1),
                                 (np.where((time >= .395) & (time < .445),440.,220.),0)):
        wave = np.sin(2*np.pi*np.cumsum(sung)/rate)
        result, report = review_pitch(wave,estimate,rate,hop)
        assert json.loads(json.dumps(report))['corrected'] == expected_count
        assert np.allclose(result[40:45],220. if expected_count else 440.)
