"""Fine-tune a condition adapter against the pinned TIMotion diffusion loss.

This entrypoint reuses upstream preprocessing and loss, with a frozen pretrained
denoiser. Weak keyword presence is NOT an attack-role or contact annotation.
No test-source files are opened. Downstream utility still needs tracking runs.
"""
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
from combat.paired_models import CombatConditionAdapter
from combat.protocol import check_artifact, load_manifest

TAGS = ('boxing', 'defense', 'kick', 'martial_arts', 'punch')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--validate-every', type=int, default=100)
    args = parser.parse_args()
    checkpoint, manifest, output = (p.resolve() for p in (args.checkpoint, args.manifest, args.output))
    if output.exists(): raise FileExistsError(output)
    if not 15 <= args.frames <= 300 or min(args.steps,args.batch_size,args.validate_every) < 1:
        raise ValueError('Invalid run budget')
    records = load_manifest(manifest)
    train = [r for r in records if r['split'] == 'train' and r.get('weak_tags')]
    val = [r for r in records if r['split'] == 'val' and r.get('weak_tags')]
    if len(train) < 20 or len(val) < 5:
        raise ValueError('Need >=20 training and >=5 validation paired combat candidate sources')
    root = Path(__file__).resolve().parents[1]
    import clip
    clip.load = functools.partial(clip.load, download_root=str(root/'cache/clip'))
    os.chdir(root/'third_party/TIMotion')
    from configs import get_config
    from models import TIMotion
    from utils.preprocess import load_motion
    from utils.utils import process_motion_np, qmul_np, qinv_np, qrot_np, rigid_transform
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    cfg = get_config('configs/model.yaml')
    cfg.NUM_HEADS = 16; cfg.NUM_LAYERS = 5; cfg.LATENT_DIM = 512; cfg.LPA = False
    model = TIMotion(cfg)
    from combat.checkpoint_io import load_model_state
    state, checkpoint_hash = load_model_state(checkpoint)
    model.load_state_dict(state, strict=True)
    del state
    model = model.cuda().eval().requires_grad_(False)
    adapter = CombatConditionAdapter({tag:2 for tag in TAGS}).cuda()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=1e-4)
    cache = {}
    for record in train+val:
        check_artifact(record['annotation_path'],record['annotation_sha256'])
        paths = record['motion_paths']
        arrays = [load_motion(p, 16, swap=False)[0] for p in paths]
        if any(a is None for a in arrays) or len(arrays[0]) != len(arrays[1]):
            raise ValueError('Missing, too short or unsynchronized source: '+record['sample_id'])
        texts = [t.strip() for t in Path(record['annotation_path']).read_text(encoding='utf-8').splitlines() if t.strip()]
        if not texts: raise ValueError('No official annotation')
        cache[record['sample_id']] = (arrays,texts)

    def batch(rows, rng):
        motions, lengths, texts = [], [], []
        for record in rows:
            arrays, captions = cache[record['sample_id']]
            size = min(len(arrays[0]), args.frames+1)
            start = rng.randrange(len(arrays[0])-size+1)
            # Copies prevent upstream in-place coordinate transforms mutating cache.
            a, qa, pa = process_motion_np(arrays[0][start:start+size].copy(), .001, 0, 22)
            b, qb, pb = process_motion_np(arrays[1][start:start+size].copy(), .001, 0, 22)
            relative = qmul_np(qb, qinv_np(qa))
            angle = np.arctan2(relative[:,2:3],relative[:,0:1])
            xz = qrot_np(qa,pb-pa)[:,[0,2]]
            b = rigid_transform(np.concatenate([angle,xz],axis=-1)[0],b)
            pair = np.concatenate([a,b],axis=-1)
            lengths.append(len(pair))
            motions.append(np.pad(pair,((0,args.frames-len(pair)),(0,0))))
            texts.append(rng.choice(captions))
        data = {'text':texts,'motions':torch.tensor(np.stack(motions),device='cuda',dtype=torch.float32),
                'motion_lens':torch.tensor(lengths,device='cuda',dtype=torch.long)}
        with torch.no_grad(): data = model.text_process(data)
        conditions = {tag:torch.tensor([int(tag in r['weak_tags']) for r in rows],device='cuda') for tag in TAGS}
        data['cond'] = adapter(data['cond'],conditions)
        return data

    output.mkdir(parents=True)
    provenance = {'seed':args.seed,'steps':args.steps,'batch_size':args.batch_size,'frames':args.frames,
        'model':'official TIMotion Transformer no-LPA, frozen; trainable condition adapter',
        'condition_semantics':'weak keyword presence only; no verified attack/defender/contact roles',
        'upstream_commit':'172580fd7f2551a7cc10fde1f15ff73315742e65',
        'checkpoint_sha256':checkpoint_hash,
        'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'training_sources':[r['source_ids'] for r in train], 'validation_sources':[r['source_ids'] for r in val],
        'torch':torch.__version__, 'trainable_parameters':sum(p.numel() for p in adapter.parameters())}
    (output/'config.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    rng=random.Random(args.seed);best=float('inf');begin=time.monotonic()
    with (output/'loss.jsonl').open('w',encoding='utf-8') as log:
        for step in range(1,args.steps+1):
            adapter.train();rows=rng.choices(train,k=args.batch_size)
            optimizer.zero_grad()
            loss=model.decoder.compute_loss(batch(rows,rng))['total']
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite diffusion loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),.5);optimizer.step()
            item={'step':step,'loss':float(loss.detach()),'elapsed_seconds':time.monotonic()-begin}
            if step % args.validate_every == 0 or step == args.steps:
                adapter.eval(); values=[]
                # Same noise/mask/crops each checkpoint; restore training RNG state.
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(1729);vrng=random.Random(1729)
                    with torch.no_grad():
                        for index in range(0,min(len(val),32),args.batch_size):
                            values.append(float(model.decoder.compute_loss(batch(val[index:index+args.batch_size],vrng))['total']))
                item['validation_diffusion_loss']=float(np.mean(values))
                if item['validation_diffusion_loss'] < best:
                    best=item['validation_diffusion_loss']
                    torch.save({'adapter':adapter.state_dict(),'cardinalities':{t:2 for t in TAGS},
                                'step':step,'provenance':provenance},output/'best_adapter.pt')
            log.write(json.dumps(item)+'\n');log.flush()
            if step==1 or step%10==0: print(json.dumps(item),flush=True)
    torch.save({'adapter':adapter.state_dict(),'optimizer':optimizer.state_dict(),'step':args.steps,
                'provenance':provenance},output/'last_adapter.pt')
    (output/'TRAINING_COMPLETE.json').write_text(json.dumps({'steps':args.steps,'best_validation_loss':best,
        'official_generation_metrics_run':False,'tracking_utility_verified':False},indent=2),encoding='utf-8')


if __name__ == '__main__': main()
