import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from combat.benchmark_readiness import check_manifest


class BenchmarkReadiness(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'motion.bin').write_bytes(b'fixture-not-research-data')
        (self.root/'review.txt').write_text('fixture')
        self.record = dict(sample_id='test1', dataset='fixture', split='test', source_ids=['capture1'],
            lineage_ids=[], source_sha256=hashlib.sha256(b'fixture-not-research-data').hexdigest(),
            motion_path='motion.bin', representation='fixture', fps=30, license_status='verified',
            test_use='confirmation', untouched=True, independent_annotation=True, annotation_id='event1')
        self.label = dict(annotation_id='event1', sample_id='test1', split='test', event_kind='Block',
            independent_of_generation_loss='true', annotator_1='r1', annotator_2='r2',
            agreement='agreed', evidence_type='manual_review', evidence_path='review.txt',
            start_s='0', window_start_s='1', window_end_s='2', end_s='3')

    def check(self, records=None):
        manifest=self.root/'manifest.jsonl'; annotations=self.root/'ann.csv'
        manifest.write_text('\n'.join(json.dumps(x) for x in (records if records is not None else [self.record])))
        with annotations.open('w', newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(self.label));writer.writeheader();writer.writerow(self.label)
        return check_manifest(self.root, manifest, annotations)

    def test_complete_provenance(self):
        self.assertTrue(self.check()['confirmation_manifest_ready'])

    def test_empty_not_ready(self):
        self.assertFalse(self.check([])['confirmation_manifest_ready'])

    def test_descendant_cross_split(self):
        train=dict(self.record,sample_id='train',split='train',source_ids=['derived'],lineage_ids=['capture1'])
        self.assertFalse(self.check([train,self.record])['confirmation_manifest_ready'])

    def test_development_cannot_be_confirmation(self):
        self.record['untouched']=False
        self.assertFalse(self.check()['confirmation_manifest_ready'])

    def test_pseudo_label_rejected(self):
        self.label['evidence_type']='geometry_pseudo'
        self.assertFalse(self.check()['confirmation_manifest_ready'])

    def test_future_test_condition_for_training_rejected(self):
        r=dict(self.record,sample_id='train',split='train',source_ids=['train1'],synthetic=True,conditioning_split='test')
        self.assertFalse(self.check([r,self.record])['confirmation_manifest_ready'])

    def test_missing_file_and_corruption_rejected(self):
        (self.root/'motion.bin').write_bytes(b'changed')
        self.assertFalse(self.check()['confirmation_manifest_ready'])
        (self.root/'motion.bin').unlink()
        self.assertFalse(self.check()['confirmation_manifest_ready'])

    def test_unresolved_review_and_time_rejected(self):
        self.label['agreement']='disagreed';self.label['window_end_s']='9'
        self.assertFalse(self.check()['confirmation_manifest_ready'])


if __name__=='__main__':unittest.main()
