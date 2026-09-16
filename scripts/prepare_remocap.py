import argparse
from pathlib import Path
from combat.remocap_data import prepare

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args()
    prepare(a.root,a.output,a.receipt)
