"""Run pinned upstream evaluation functions with audited offline I/O adaptations.

No evaluator algorithm is replaced. Dataset package initialization is bypassed
to avoid importing the unused Lightning training DataModule. Only the two known
legacy annotations receive their declared decoding. This is a reduced-repeat
replication on available official test files, not a paper-table reproduction.
"""
import argparse
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import types
import numpy as np
import torch


def module_from_file(name,path,text=None):
    module=types.ModuleType(name);module.__file__=str(path)
    sys.modules[name]=module
    exec(compile(path.read_text() if text is None else text,str(path),'exec'),module.__dict__)
    return module


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--mm-samples',type=int,default=100)
    p.add_argument('--mm-repeats',type=int,default=30)
    a=p.parse_args();out=a.output.resolve()
    if out.exists():raise FileExistsError(out)
    if a.repeats<1 or a.mm_samples<11 or a.mm_repeats<11:raise ValueError('Insufficient metric budget')
    root=Path(__file__).resolve().parents[1];up=root/'third_party/TIMotion'
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    import clip
    clip.load=functools.partial(clip.load,download_root=str(root/'cache/clip'))
    os.chdir(up)
    package=types.ModuleType('datasets');package.__path__=[str(up/'datasets')]
    sys.modules['datasets']=package
    source=up/'datasets/interhuman.py';original=source.read_text()
    old='open(text_path, "r").readlines()'
    if original.count(old)!=1:raise RuntimeError('Upstream dataset source drift')
    adapted='from combat.annotations import read_annotation\n'+original.replace(old,'read_annotation(text_path)[0].splitlines(keepends=True)')
    dataset=module_from_file('datasets.interhuman',source,adapted)
    package.InterHumanDataset=dataset.InterHumanDataset
    evaluator=module_from_file('datasets.evaluator',up/'datasets/evaluator.py')
    from combat.checkpoint_io import load_model_state
    def verified_evaluator(cfg):
        model=evaluator.InterCLIP(cfg)
        state,sha=load_model_state(root/'data/official/interhuman_evaluator_weights.npz')
        model.load_state_dict(state,strict=True)
        print('EVALUATOR_STRICT_LOAD '+sha,flush=True)
        return model
    evaluator.build_models=verified_evaluator
    for name in ('EvaluatorModelWrapper','EvaluationDataset','get_dataset_motion_loader','get_motion_loader'):
        setattr(package,name,getattr(evaluator,name))
    evaluation=module_from_file('offline_upstream_eval',up/'tools/eval.py')
    cfg=evaluation.get_config('configs/model.yaml')
    cfg.NUM_HEADS=16;cfg.NUM_LAYERS=5;cfg.LATENT_DIM=512;cfg.LPA=False
    model=evaluation.build_models(cfg)
    state,sha=load_model_state(root/'data/official/timotion_weights.npz')
    model.load_state_dict(state,strict=True);del state
    device=torch.device('cuda');model=model.to(device).eval()
    print('GENERATOR_STRICT_LOAD '+sha,flush=True)
    data_cfg=evaluation.get_config('configs/datasets.yaml').interhuman_test
    data_cfg.DATA_ROOT=str(root/'data/interhuman')
    loader,data=evaluator.get_dataset_motion_loader(data_cfg,96)
    evaluation.mm_num_times=10;evaluation.diversity_times=300
    evaluation.replication_times=a.repeats
    evaluation.gt_loader=loader;evaluation.gt_dataset=data
    evaluation.eval_wrapper=evaluator.EvaluatorModelWrapper(evaluation.get_config('configs/eval_model.yaml'),device)
    evaluation.eval_motion_loaders={cfg.NAME:lambda:evaluator.get_motion_loader(96,model,data,device,a.mm_samples,a.mm_repeats)}
    out.mkdir(parents=True)
    provenance={'seed':a.seed,'repeats':a.repeats,'paper_default_repeats':20,
        'batch_size':96,'mm_samples':a.mm_samples,'mm_repeats':a.mm_repeats,
        'test_samples_loaded':len(data),'raw_test_ids':1177,
        'missing_original_motion_id':'4106','short_sequences_skipped_by_upstream':True,
        'annotation_source':'mirror; 50/50 official byte comparisons, remainder not individually verified',
        'adaptations':['verified array-only model transport; strict key loading',
            'bypass unused Lightning DataModule initialization',
            'GB18030 decode only original annotation IDs 4642 and 5834'],
        'identity_postprocess':False,'contact_guidance':False,
        'upstream_commit':'172580fd7f2551a7cc10fde1f15ff73315742e65',
        'source_sha256':{str(path.relative_to(up)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [source,up/'datasets/evaluator.py',up/'tools/eval.py']},
        'claim':'upstream metrics on available test subset, reduced repetitions; not full published benchmark replication'}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    evaluation.evaluation(str(out/'evaluation.log'))
    (out/'EVALUATION_COMPLETE.json').write_text(json.dumps({'complete':True,'full_paper_protocol':False}))


if __name__=='__main__':main()
