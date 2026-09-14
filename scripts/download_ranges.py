"""Bounded public-file range downloader with resumable chunks and final SHA256.

Only use a source URL and exact total length verified from the publisher's HTTP
Content-Range. Optional prefix must be a stopped, partial download of SAME URL.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--url',required=True)
    p.add_argument('--bytes',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prefix',type=Path)
    p.add_argument('--workers',type=int,default=3)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    start=args.prefix.stat().st_size if args.prefix else 0
    if start>=args.bytes:raise ValueError('Invalid prefix length')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(args.output.parent).free < 2*args.bytes-start+3*1024**3:
        raise RuntimeError('Insufficient space for chunks, assembly and 3GiB reserve')
    parts=args.output.with_suffix(args.output.suffix+'.chunks');parts.mkdir(exist_ok=True)
    spec={'url':args.url,'bytes':args.bytes,'prefix_bytes':start}
    spec_path=parts/'source.json'
    if spec_path.exists() and json.loads(spec_path.read_text())!=spec:
        raise ValueError('Chunk directory belongs to a different source or prefix')
    spec_path.write_text(json.dumps(spec,indent=2))
    chunk_size=32*1024**2
    spans=[(a,min(a+chunk_size,args.bytes)-1) for a in range(start,args.bytes,chunk_size)]
    curl='curl.exe' if os.name=='nt' else 'curl'
    def fetch(span):
        a,b=span;dest=parts/f'{a:012d}.bin';header=parts/f'{a:012d}.headers'
        if dest.exists() and dest.stat().st_size==b-a+1 and header.exists():
            content=header.read_text(errors='replace')
            if re.search(rf'(?im)^content-range:\s*bytes {a}-{b}/{args.bytes}\s*$',content):return dest
        cmd=[curl,'-sS','--fail','-L','--connect-timeout','15','--max-time','600','--retry','2','--retry-all-errors','--retry-delay','2',
             '--range',f'{a}-{b}','-D',str(header),'-o',str(dest),args.url]
        subprocess.run(cmd,check=True,timeout=1850)
        content=header.read_text(errors='replace')
        if dest.stat().st_size!=b-a+1 or not re.search(rf'(?im)^content-range:\s*bytes {a}-{b}/{args.bytes}\s*$',content):
            raise RuntimeError('Wrong response range for '+str(dest))
        print(json.dumps({'chunk_start':a,'bytes':b-a+1}),flush=True)
        return dest
    with ThreadPoolExecutor(max_workers=max(1,min(args.workers,4))) as pool:
        downloaded=list(pool.map(fetch,spans))
    assembled=args.output.with_suffix(args.output.suffix+'.assembling')
    digest=hashlib.sha256();written=0
    with assembled.open('wb') as f:
        for source in ([args.prefix] if args.prefix else [])+downloaded:
            with source.open('rb') as src:
                for block in iter(lambda:src.read(1024**2),b''):
                    f.write(block);digest.update(block);written+=len(block)
    if written!=args.bytes:raise RuntimeError('Wrong assembled size')
    assembled.rename(args.output)
    result={**spec,'sha256':digest.hexdigest(),'status':'complete'}
    args.output.with_suffix(args.output.suffix+'.download.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
