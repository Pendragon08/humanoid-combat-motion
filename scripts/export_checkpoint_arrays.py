"""Export official PyTorch state_dict tensors without importing/executing Torch.

Restricted reader for this audited float32 checkpoint, not a general unpickler.
Optimizer tensors are omitted. Each parameter and the input archive are hashed;
server inference must still strict-load every model state_dict key.
"""
import argparse
import codecs
import collections
import hashlib
import io
import json
import pickle
from pathlib import Path
import zipfile
import numpy as np


class FloatStorage: pass


def tensor(storage, offset, size, stride, requires_grad, hooks):
    return {'storage':storage,'offset':offset,'size':size,'stride':stride}


class Reader(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('collections','OrderedDict'):collections.OrderedDict,
                 ('torch','FloatStorage'):FloatStorage,
                 ('torch._utils','_rebuild_tensor_v2'):tensor,
                 ('_codecs','encode'):codecs.encode,('numpy','dtype'):np.dtype,
                 ('numpy.core.multiarray','scalar'):np.core.multiarray.scalar}
        if (module,name) not in allowed:raise ValueError('Unsupported pickle global: '+module+'.'+name)
        return allowed[(module,name)]

    def persistent_load(self,item):
        if len(item)!=5 or item[0]!='storage' or item[1] is not FloatStorage:
            raise ValueError('Unsupported storage descriptor')
        return {'key':str(item[2]),'elements':int(item[4])}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    records=[]
    with zipfile.ZipFile(a.input) as source, zipfile.ZipFile(a.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as dest:
        meta=next(n for n in source.namelist() if n.endswith('/data.pkl'))
        prefix=meta.rsplit('/',1)[0]
        checkpoint=Reader(io.BytesIO(source.read(meta))).load()
        for name,desc in checkpoint['state_dict'].items():
            if not name.startswith('model.') or not isinstance(desc,dict):raise ValueError('Unexpected state entry')
            storage=desc['storage'];raw=source.read(prefix+'/data/'+storage['key'])
            if len(raw)!=4*storage['elements']:raise ValueError('Storage length mismatch')
            size=tuple(map(int,desc['size']));stride=tuple(map(int,desc['stride']));offset=int(desc['offset'])
            if len(size)!=len(stride) or any(v<0 for v in size+stride) or offset<0:raise ValueError('Invalid view')
            last=offset+sum((n-1)*s for n,s in zip(size,stride) if n)
            if last>=storage['elements']:raise ValueError('View exceeds storage')
            view=np.ndarray(size,dtype='<f4',buffer=raw,offset=offset*4,strides=tuple(s*4 for s in stride)).copy()
            if not np.isfinite(view).all():raise ValueError('Nonfinite tensor')
            stream=io.BytesIO();np.save(stream,view,allow_pickle=False)
            key=name.removeprefix('model.')
            dest.writestr(key+'.npy',stream.getvalue())
            records.append({'key':key,'shape':list(size),'sha256':hashlib.sha256(view.tobytes()).hexdigest()})
    record={'source_sha256':sha(a.input),'export_sha256':sha(a.output),'parameters':records,
            'format':'numpy state_dict, float32, optimizer omitted','source_bytes':a.input.stat().st_size,'export_bytes':a.output.stat().st_size}
    a.output.with_suffix('.manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in record.items() if k!='parameters'}),flush=True)
    print('tensor_count',len(records),flush=True)


if __name__=='__main__':main()
