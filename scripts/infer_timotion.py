"""Strict-load the official Transformer checkpoint; save paired motions + provenance."""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import clip


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--prompts',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--frames',type=int,default=150)
    p.add_argument('--heads',type=int,default=16)
    p.add_argument('--layers',type=int,default=5)
    p.add_argument('--latent',type=int,default=512)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    checkpoint=args.checkpoint.resolve(); prompts=args.prompts.resolve(); output=args.output.resolve()
    if checkpoint.suffix=='.partial' or output.exists():
        raise ValueError('Incomplete checkpoint or output already exists')
    if not 15<=args.frames<=300:
        raise ValueError('Use the supported 15..300-frame range')
    texts=json.loads(prompts.read_text(encoding='utf-8'))
    clip.load=functools.partial(clip.load,download_root=str(root/'cache/clip'))
    os.chdir(root/'third_party/TIMotion')
    from configs import get_config
    from models import TIMotion
    from utils.utils import MotionNormalizer
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark=False
    cfg=get_config('configs/model.yaml')
    cfg.NUM_HEADS=args.heads; cfg.NUM_LAYERS=args.layers; cfg.LATENT_DIM=args.latent; cfg.LPA=False
    model=TIMotion(cfg)
    checkpoint_hash=sha(checkpoint)
    # The user-authorized official Lightning checkpoint includes training metadata.
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    raw=state['state_dict']
    keys={k.removeprefix('model.'):v for k,v in raw.items()}
    model.load_state_dict(keys,strict=True)
    del state,raw,keys
    model=model.cuda().eval()
    normalizer=MotionNormalizer()
    output.mkdir(parents=True)
    records=[]
    for index,text in enumerate(texts):
        start=time.monotonic()
        with torch.no_grad():
            batch=model.forward_test({'text':[text],'motion_lens':torch.tensor([args.frames],device='cuda')})
        normalized=batch['output'].reshape(1,args.frames,2,262).cpu().numpy()
        features=normalizer.backward(normalized)[0].astype(np.float32)
        if not np.isfinite(features).all():raise FloatingPointError('Nonfinite generated motion')
        positions=features[...,:66].reshape(args.frames,2,22,3)
        target=output/f'sample_{index:03d}.npz'
        np.savez_compressed(target,features=features,positions=positions,fps=30)
        records.append({'sample_id':target.stem,'text':text,'seed':args.seed,'seconds':time.monotonic()-start,
                        'sha256':sha(target),'motion_kind':'synthetic_human','coordinate_system':'upstream_interhuman_y_up',
                        'robot_ready':False,'official_benchmark':False})
        print(json.dumps(records[-1]),flush=True)
    report={'checkpoint_sha256':checkpoint_hash,'source_commit':'172580fd7f2551a7cc10fde1f15ff73315742e65',
            'architecture':'TIMotion-Transformer-no-LPA','cfg':dict(heads=args.heads,layers=args.layers,latent=args.latent),
            'strict_load':True,'torch':torch.__version__,'seed':args.seed,'frames':args.frames,'samples':records,
            'claim':'qualitative inference only; not official FID or robot tracking evaluation'}
    (output/'manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')


if __name__=='__main__':main()
