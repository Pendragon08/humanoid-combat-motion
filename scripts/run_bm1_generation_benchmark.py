"""Pinned upstream metrics, paired RNG and a frozen caption-only adapter comparison.

This is BM1 on the available official InterHuman test subset. It is neither an
independent combat-contact test nor a downstream robot training experiment.
"""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import types

import numpy as np
import torch
from combat.benchmark_conditions import caption_conditions, assert_no_test_overlap
from combat.checkpoint_io import load_model_state, sha256
from scripts.evaluate_timotion_offline import module_from_file


def reset_rng(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=['baseline', 'caption_adapter'], required=True)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--seed', type=int, default=20260915)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--smoke-only', action='store_true')
    a = p.parse_args(); root = Path(__file__).resolve().parents[1]
    out = a.output.resolve(); ap = a.adapter.resolve() if a.adapter else None
    if out.exists(): raise FileExistsError(out)
    if a.repeats < 1 or (a.arm == 'caption_adapter') != (ap is not None):
        raise ValueError('Invalid arm/checkpoint/budget')
    up = root / 'third_party/TIMotion'
    pins = {'datasets/interhuman.py': '58f0426053dc28c3d7326205be6f0ac977609eed826be3804e7739cefe43f954',
            'datasets/evaluator.py': '37dc066ee72200716f4119d1261d135e669c8c448a57fa6705ac029904f0ff57',
            'tools/eval.py': '627ee96e7422d775091ab4c88c13396c4727b0a3baf648611c7a4e0cae87abc5'}
    for name, digest in pins.items():
        if sha256(up/name) != digest: raise ValueError('Upstream source drift: ' + name)
    spec = json.dumps(json.loads((root/'env-timotion.json').read_text()), sort_keys=True, separators=(',', ':'))
    if hashlib.sha256(spec.encode()).hexdigest()[:8] != 'bbd2963d': raise ValueError('Environment specification changed')
    import clip
    clip.load = functools.partial(clip.load, download_root=str(root/'cache/clip'))
    os.chdir(up); reset_rng(a.seed); torch.backends.cudnn.benchmark = False
    package = types.ModuleType('datasets'); package.__path__ = [str(up/'datasets')]; sys.modules['datasets'] = package
    source = up/'datasets/interhuman.py'; text = source.read_text()
    needle = 'open(text_path, "r").readlines()'
    if text.count(needle) != 1: raise ValueError('Dataset adaptation drift')
    ds = module_from_file('datasets.interhuman', source,
                          'from combat.annotations import read_annotation\n' + text.replace(needle, 'read_annotation(text_path)[0].splitlines(keepends=True)'))
    package.InterHumanDataset = ds.InterHumanDataset
    evaluator = module_from_file('datasets.evaluator', up/'datasets/evaluator.py')
    def load_evaluator(cfg):
        m = evaluator.InterCLIP(cfg); state, digest = load_model_state(root/'data/official/interhuman_evaluator_weights.npz')
        m.load_state_dict(state, strict=True); print('EVALUATOR_STRICT_LOAD', digest, flush=True); return m
    evaluator.build_models = load_evaluator
    for name in ('EvaluatorModelWrapper', 'EvaluationDataset', 'get_dataset_motion_loader', 'get_motion_loader'):
        setattr(package, name, getattr(evaluator, name))
    ev = module_from_file('bm1_upstream_eval', up/'tools/eval.py')
    cfg = ev.get_config('configs/model.yaml'); cfg.NUM_HEADS=16; cfg.NUM_LAYERS=5; cfg.LATENT_DIM=512; cfg.LPA=False
    model = ev.build_models(cfg); state, backbone_sha = load_model_state(root/'data/official/timotion_weights.npz')
    model.load_state_dict(state, strict=True); del state; model = model.cuda().eval()
    print('GENERATOR_STRICT_LOAD', backbone_sha, flush=True)
    data_cfg = ev.get_config('configs/datasets.yaml').interhuman_test; data_cfg.DATA_ROOT = str(root/'data/interhuman')
    loader, data = evaluator.get_dataset_motion_loader(data_cfg, 96)
    names = [r['name'] for r in data.data_list]
    adapter = None; adapter_provenance = None
    if ap:
        from combat.paired_models import CombatConditionAdapter
        saved = torch.load(ap, map_location='cpu', weights_only=False)
        if saved['provenance']['checkpoint_sha256'] != backbone_sha: raise ValueError('Adapter backbone mismatch')
        assert_no_test_overlap(saved['provenance'], names)
        adapter = CombatConditionAdapter(saved['cardinalities']).cuda().eval()
        adapter.load_state_dict(saved['adapter'], strict=True); adapter_provenance = saved['provenance']
    ev.eval_wrapper = evaluator.EvaluatorModelWrapper(ev.get_config('configs/eval_model.yaml'), torch.device('cuda'))
    # Reset after every model, evaluator and adapter initialization; both arms
    # then receive the same captions, crops, shuffles, length and diffusion RNG.
    reset_rng(a.seed)
    out.mkdir(parents=True)
    all_ids = (root/'data/interhuman/test.txt').read_text().split()
    inputs = []
    for name in names:
        paths = [root/'data/interhuman/motions_processed'/person/(name+'.npy') for person in ('person1', 'person2')]
        paths.append(root/'data/interhuman/annots'/(name+'.txt'))
        inputs.append({'id': name, 'files': {str(q.relative_to(root)):sha256(q) for q in paths}})
    save(out/'frozen_inputs.json', inputs)
    provenance = {'arm':a.arm,'seed':a.seed,'repetitions':a.repeats,'smoke_only':a.smoke_only,
        'test_count':len(names),'test_order':names,'raw_test_ids':len(all_ids),
        'unloaded_ids':sorted(set(all_ids)-set(names)),
        'frozen_inputs_sha256':sha256(out/'frozen_inputs.json'),
        'official_split_sha256':sha256(root/'data/interhuman/test.txt'),
        'backbone_original_sha256':backbone_sha,'adapter_sha256':sha256(ap) if ap else None,
        'adapter_training_provenance':adapter_provenance,
        'adapter_selection':'seed0 chosen using existing validation losses only; no test selection',
        'conditioning':'caption and length only; weak tags derived from the same selected caption, no motion/contact labels',
        'training_note':'original adapter was trained with tags pooled across source captions; this evaluation uses current caption only',
        'metrics':'unaltered pinned upstream InterCLIP/FID/R-precision/diversity/multimodality functions',
        'batch_size':96,'mm_samples':100,'mm_repeats':30,'diversity_times':300,
        'subset_note':'missing raw source and upstream short-sequence exclusions; not the complete published dataset',
        'annotation_note':'mirror annotations; earlier 50/50 official byte spot-check, not full provenance certification',
        'upstream_sha256':pins,'runner_sha256':sha256(Path(__file__)),
        'condition_module_sha256':sha256(root/'combat/benchmark_conditions.py'),
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'new_training_steps':0,'combat_contact_verified':False,'robot_utility_verified':False}
    save(out/'provenance.json', provenance)
    original_forward = model.forward_test; count=0; begin=time.monotonic()
    trace = (out/'generation_inputs.jsonl').open('w', encoding='utf8')
    def forward(batch):
        nonlocal count
        texts = list(batch['text']); lengths = batch['motion_lens'].tolist()
        item={'call':count,'text_sha256':[hashlib.sha256(t.encode()).hexdigest()for t in texts],
              'lengths':lengths,'cuda_rng_sha256':hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()}
        if adapter is None: result=original_forward(batch)
        else:
            batch=model.text_process(batch); tags=[caption_conditions(t)for t in texts]
            conditions={k:torch.tensor([r[k]for r in tags],device='cuda',dtype=torch.long)for k in adapter.names}
            batch['cond']=adapter(batch['cond'],conditions); result=model.decode_motion(batch)
        if not torch.isfinite(result['output']).all(): raise FloatingPointError('Nonfinite generated motion')
        trace.write(json.dumps(item)+'\n');trace.flush();count+=1
        if count==1 or count%25==0: print('BM1_GENERATION_PROGRESS',count,'elapsed_s',round(time.monotonic()-begin,2),flush=True)
        return result
    model.forward_test = forward
    try:
        if a.smoke_only:
            with torch.no_grad(): forward({'text':['Two people box and one blocks a punch.'],'motion_lens':torch.tensor([30],device='cuda')})
            save(out/'SMOKE_COMPLETE.json',{'finite_generation':True,'calls':count,'metric_run':False})
            print('BM1_SMOKE_COMPLETE',flush=True); return
        ev.mm_num_times=10;ev.diversity_times=300;ev.replication_times=a.repeats
        ev.gt_loader=loader;ev.gt_dataset=data
        ev.eval_motion_loaders={a.arm:lambda:evaluator.get_motion_loader(96,model,data,torch.device('cuda'),100,30)}
        ev.evaluation(str(out/'evaluation.log'))
        save(out/'EVALUATION_COMPLETE.json',{'complete':True,'repetitions':a.repeats,'calls':count,
             'seconds':time.monotonic()-begin,'log_sha256':sha256(out/'evaluation.log'),
             'trace_sha256':sha256(out/'generation_inputs.jsonl'),'full_published_dataset':False})
        print('BM1_EVALUATION_COMPLETE',count,flush=True)
    finally: trace.close()


if __name__=='__main__': main()
