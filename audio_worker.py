"""CPU music workers. Each invocation exits to release its models and memory."""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')

def save(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def separate(args):
    import numpy as np
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    torch.set_num_threads(2)
    # Demucs 4 checkpoints contain training metadata; only load the verified official cache.
    checkpoint=Path(os.environ['TORCH_HOME'])/'hub/checkpoints/955717e8-8726e21a.th'
    if not checkpoint.exists() or not hashlib.sha256(checkpoint.read_bytes()).hexdigest().startswith('8726e21a'):
        raise ValueError('Official Demucs checkpoint missing or hash mismatch')
    original_load=torch.load
    def trusted_load(*a,**kw):
        kw['weights_only']=False
        return original_load(*a,**kw)
    torch.load=trusted_load
    model=get_model('htdemucs').cpu().eval()
    wave,rate=sf.read(args.input,dtype='float32',always_2d=True)
    if rate != model.samplerate:
        raise ValueError(f'Demucs expects {model.samplerate}Hz, received {rate}')
    if wave.shape[1]==1:
        wave=np.repeat(wave,2,axis=1)
    tensor=torch.from_numpy(wave.T)
    reference=tensor.mean(0)
    mean,std=reference.mean(),reference.std().clamp_min(1e-6)
    with torch.inference_mode():
        result=apply_model(model,((tensor-mean)/std)[None],device='cpu',shifts=0,split=True,overlap=.25,progress=False,num_workers=0)[0]
    vocals=(result[model.sources.index('vocals')]*std+mean).mean(0).numpy()
    if not np.isfinite(vocals).all() or np.sqrt(np.mean(vocals**2)) < 1e-5:
        raise ValueError('Vocal separation produced invalid or silent audio')
    sf.write(args.output,vocals,rate,subtype='PCM_16')
    save(Path(args.output).with_suffix('.json'),{'model':'htdemucs','device':'cpu','seconds':len(vocals)/rate,'sample_rate':rate})

def extract(args):
    import librosa
    import numpy as np
    import torch
    import yaml
    torch.set_num_threads(2)
    repo=Path(args.repo).resolve()
    sys.path.insert(0,str(repo))
    config=yaml.safe_load(Path(args.model).with_name('config.yaml').read_text())
    from inference import MIDIExtractionInference
    from utils.infer_utils import build_midi_file
    model=MIDIExtractionInference(config,Path(args.model),device='cpu')
    wave,_=librosa.load(args.input,sr=config['audio_sample_rate'],mono=True)
    result=model.infer([wave])[0]
    midi=build_midi_file([0.0],[result],tempo=120)
    midi.save(args.output)
    cursor=0.0
    notes=[]
    for pitch,duration,rest in zip(result['note_midi'],result['note_dur'],result['note_rest']):
        end=cursor+float(duration)
        if not bool(rest) and end-cursor >= .04:
            notes.append({'start':cursor,'end':end,'pitch':float(pitch)})
        cursor=end
    if not notes:
        raise ValueError('SOME found no sung notes')
    save(Path(args.output).with_suffix('.notes.json'),{'model':'openvpi/SOME v1.0.0-baseline','device':'cpu','notes':notes,'duration':cursor})

def synthesize(args):
    import numpy as np
    import onnxruntime as ort
    import torch
    torch.set_num_threads(2)
    original_session=ort.InferenceSession
    def bounded_session(path,sess_options=None,**kwargs):
        options=sess_options or ort.SessionOptions()
        options.intra_op_num_threads=2
        options.inter_op_num_threads=1
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        kwargs['providers']=['CPUExecutionProvider']
        return original_session(path,sess_options=options,**kwargs)
    ort.InferenceSession=bounded_session
    from diffsinger_utau.voice_bank import PredAll
    from diffsinger_utau.voice_bank.commons.ds_reader import DSReader
    ds=DSReader(Path(args.input)).read_ds()[0]
    predictor=PredAll(Path(args.model))
    # PredAll otherwise re-predicts ph_dur even if provided. Preserve SOFA's
    # measured articulation, especially short consonants before sustained vowels.
    if 'ph_dur' not in ds:
        raise ValueError('Measured singing phoneme durations are required')
    measured=np.asarray(ds.get_list('ph_dur'),dtype=np.float32)
    if len(measured)!=len(ds.get_list('ph_seq')) or not np.all(measured>0):
        raise ValueError('Invalid measured phoneme duration count or value')
    predictor.pred_duration.predict=lambda *a,**kw:measured.copy()
    result=predictor.predict_full_pipeline(ds=ds,lang='zh',speaker=None,pitch_steps=10,variance_steps=10,acoustic_steps=20,output_dir=str(Path(args.output).parent/'diffsinger'),save_intermediate=True)
    import shutil
    shutil.copyfile(result['audio_path'],args.output)
    import soundfile as sf
    wave,rate=sf.read(args.output,dtype='float32')
    if not np.isfinite(wave).all() or np.sqrt(np.mean(wave**2)) < 1e-5:
        raise ValueError('SVS output invalid or silent')
    save(Path(args.output).with_suffix('.json'),{'engine':'DiffSinger','voicebank':'绮萱 Qixuan 2.7.0','device':'cpu','seconds':len(wave)/rate,'sample_rate':rate,'source_attribution_required':True})

def align(args):
    import torch
    torch.set_num_threads(2)
    repo=Path(args.repo).resolve()
    sys.path.insert(0,str(repo))
    from modules.g2p.dictionary_g2p import DictionaryG2P
    from modules.task.forced_alignment import LitForcedAlignmentTask
    from pypinyin import lazy_pinyin
    import re
    text=json.loads(Path(args.text).read_text(encoding='utf-8'))['text']
    chars=re.findall(r'[\u3400-\u9fff]',text)
    syllables=lazy_pinyin(''.join(chars),strict=False)
    g2p=DictionaryG2P(dictionary=str(repo/'dictionary/opencpop-extension.txt'))
    phones,words,mapping=g2p(' '.join(syllables))
    if len(words)!=len(chars):
        raise ValueError('SOFA dictionary did not accept every lyric syllable')
    checkpoint=Path(args.model)
    manifest=json.loads(checkpoint.with_name('provenance.json').read_text(encoding='utf-8-sig'))
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=manifest['sha256']:
        raise ValueError('SOFA checkpoint hash mismatch')
    model=LitForcedAlignmentTask.load_from_checkpoint(str(checkpoint),map_location='cpu',weights_only=False).cpu().eval()
    model.set_inference_mode('force')
    with torch.inference_mode():
        model.on_predict_start()
        result=model.predict_step((Path(args.input),phones,words,mapping),0)
    _,duration,confidence,ph_seq,ph_intervals,word_seq,word_intervals=result
    if list(word_seq)!=words or len(word_intervals)!=len(chars):
        raise ValueError('Forced alignment changed lyric syllable count')
    aligned=[{'text':char,'start':float(interval[0]),'end':float(interval[1])} for char,interval in zip(chars,word_intervals)]
    previous=0.
    for item in aligned:
        if not previous-.002<=item['start']<item['end']<=duration+.002:
            raise ValueError('Invalid singing forced-alignment interval')
        previous=item['end']
    save(args.output,{'chars':aligned,'confidence':float(confidence),'model':'qiuqiao/SOFA Mandarin Singing 1.0.0','phones':[{'text':str(p),'start':float(t[0]),'end':float(t[1])} for p,t in zip(ph_seq,ph_intervals)]})

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=['separate','extract','synthesize','align'])
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--model')
    parser.add_argument('--repo')
    parser.add_argument('--text')
    args=parser.parse_args()
    began=time.monotonic()
    globals()[args.operation](args)
    print(json.dumps({'operation':args.operation,'elapsed_seconds':time.monotonic()-began}),flush=True)
