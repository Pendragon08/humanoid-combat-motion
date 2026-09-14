"""Audit official splits and source-disjoint pairs; keyword tags remain weak labels."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from combat.protocol import source_split_check


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    records=[];missing=[];split_hash={};counts={};assigned={}
    patterns={'punch':r'\bpunch(?:es|ing)?\b|\bfists?\b','kick':r'\bkick(?:s|ing)?\b',
              'boxing':r'\bbox(?:ing|ers?)\b','martial_arts':r'\bmartial arts\b|\bkung fu\b',
              'defense':r'\bdodg(?:e|es|ing)\b|\bdefen(?:d|ds|ding|se)\b|\bblock(?:s|ing)?\b'}
    for split in ('train','val','test'):
        file=args.data/f'{split}.txt'
        content=file.read_bytes();split_hash[split]=hashlib.sha256(content).hexdigest()
        ids=[s.strip() for s in content.decode().splitlines() if s.strip()]
        counts[split]={'official_ids':len(ids),'complete_pairs':0,'weak_combat_pairs':0}
        for name in ids:
            if name in assigned:raise ValueError('Repeated original source in splits: '+name)
            if not name.isdigit():raise ValueError('Unexpected source identifier: '+name)
            assigned[name]=split
            files=[args.data/'motions_processed'/person/f'{name}.npy' for person in ('person1','person2')]
            annotation=args.data/'annots'/f'{name}.txt'
            absent=[str(x) for x in files+[annotation] if not x.exists()]
            if absent:
                missing.append({'source':name,'split':split,'missing':absent});continue
            text=annotation.read_text(encoding='utf-8')
            tags=[key for key,pattern in patterns.items() if re.search(pattern,text,re.I)]
            records.append({'sample_id':'interhuman_'+name,'source_ids':['interhuman:'+name],
                            'split':split,'synthetic':False,'paired':True,'motion_paths':[str(f.resolve()) for f in files],
                            'annotation_path':str(annotation.resolve()),'annotation_sha256':hashlib.sha256(annotation.read_bytes()).hexdigest(),
                            'weak_tags':tags,'label_quality':'keyword_candidate_requires_review',
                            'attack_type':'unknown','attack_actor':'unknown','defender_actor':'unknown','contact_verified':False})
            counts[split]['complete_pairs']+=1
            counts[split]['weak_combat_pairs']+=bool(tags)
    validation=source_split_check(records)
    args.output.mkdir(parents=True)
    (args.output/'manifest.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n')
    (args.output/'missing.json').write_text(json.dumps(missing,indent=2))
    summary={'counts':counts,'official_split_sha256':split_hash,'missing_pairs':len(missing),
             'official_dataset_complete':not missing,'split_check':validation,
             'labels':'weak keywords; not jab/cross/attack-defense ground truth'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
