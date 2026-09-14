import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from combat.reference_bundle import build_bundle,FIELDS


class BundleTests(unittest.TestCase):
    def test_boundaries_and_train_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);entries=[]
            for i,n in enumerate([5,7]):
                path=root/f'{i}.npz'
                np.savez(path,**{k:np.full((n,2,3),i) for k in FIELDS},fps=50,joint_names=['a','b'],body_names=['x','y'])
                entries.append(dict(path=str(path),clip_id=str(i),source_id=str(i),split='train'))
            build_bundle(entries,root/'bundle.npz')
            with np.load(root/'bundle.npz') as z:
                np.testing.assert_array_equal(z['clip_starts'],[0,5]);np.testing.assert_array_equal(z['clip_ends'],[5,12])
                self.assertEqual(z['joint_pos'][4,0,0],0);self.assertEqual(z['joint_pos'][5,0,0],1)
            entries[1]['split']='test'
            with self.assertRaises(ValueError):build_bundle(entries,root/'bad.npz')

    def test_schema_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);entries=[]
            for i,fps in enumerate([30,50]):
                path=root/f'{i}.npz';np.savez(path,**{k:np.zeros((5,2,3)) for k in FIELDS},fps=fps,joint_names=['a'],body_names=['b'])
                entries.append(dict(path=str(path),clip_id=str(i),source_id=str(i),split='train'))
            with self.assertRaises(ValueError):build_bundle(entries,root/'bad.npz')
