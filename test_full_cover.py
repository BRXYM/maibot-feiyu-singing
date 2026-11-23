from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import soundfile as sf

from pipeline import Pipeline, digest, load, save


class FullCover(TestCase):
    def test_lyric_video_preserves_opening_even_with_old_dialogue_cache(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            save(folder/'request.json', {'song':'fixture'})
            save(folder/'dialogue-filter.json', {'input_sha256':'track','intervals':[[24.72,29.98]]})
            worker = Pipeline(folder)
            source = {'title':'测试曲目 动态歌词','sha256':'track','source_url':'https://www.bilibili.com/'}
            self.assertEqual(worker.mv_dialogue(source,folder/'original.wav',265), [])
            source['title'] += ' 含剧情对白'
            self.assertEqual(worker.mv_dialogue(source,folder/'original.wav',265), [[24.72,29.98]])

    def test_legacy_v1_chunk_is_not_reused_in_v2_cover(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            save(folder/'request.json', {'song':'fixture'})
            worker = Pipeline(folder)
            worker.report['vocal_register'] = {'semitones':0}
            source = folder/'vocals.wav'
            sf.write(source,np.zeros(8000),8000)
            part = folder/'cover-parts/1'
            part.mkdir(parents=True)
            sf.write(part/'voice.wav',np.full(8000,.1),8000)
            save(part/'verified.json',{'start':0,'end':1,'sha256':digest(part/'voice.wav')})
            output = worker.convert_full_part(source,0,1,1,1)
            self.assertEqual(load(part/'verified.json')['weights'],'v2')
            self.assertFalse(np.any(sf.read(output)[0]))

    def test_cached_stems_cover_the_entire_timeline_without_asr(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            rate = 8000
            duration = 31
            tone = .02 * np.sin(2 * np.pi * 220 * np.arange(duration*rate) / rate)
            for name in ('original.wav','backing.wav','vocals.wav'):
                sf.write(folder/name, tone, rate)
            save(folder/'request.json', {'song':'fixture','full_song':True})
            worker = Pipeline(folder)
            boundaries = []

            def convert(vocals, start, end, length, index):
                boundaries.append((start,end))
                path = folder/f'part-{index}.wav'
                sf.write(path,tone[round(start*rate):round(end*rate)],rate)
                return path

            with (patch.object(worker,'voice_slot',return_value=nullcontext()),
                  patch.object(worker,'convert_full_part',side_effect=convert),
                  patch.object(worker,'helper') as separate):
                worker.run_full_song({},folder/'original.wav')
            separate.assert_not_called()
            self.assertEqual(boundaries,[(0,15.5),(15.5,31)])
            report = load(folder/'result.json')
            self.assertEqual(report['completed_parts'],2)
            self.assertTrue(report['backing'])
            self.assertEqual(report['audio_sha256'],digest(report['audio_path']))
