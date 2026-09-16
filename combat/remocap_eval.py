"""Metrics and actor-only train-library contact proposals; no test-target input."""
import numpy as np
from combat.remocap_data import PARENTS
from combat.interaction import ContactEvent

LIMBS = [20,21,10,11]
TARGETS = [9,15,18,19,20,21]
RADII = np.full(22,.055)
RADII[9]=.15; RADII[15]=.1; RADII[[20,21]]=.05; RADII[[10,11]]=.07


def actor_descriptor(actor):
    # Canonical shared frame; deliberately no responding person's data.
    ids=np.linspace(0,len(actor)-1,10).astype(int)
    return actor[ids].reshape(-1)


class TrainContactLibrary:
    """Weak retrieval baseline, NOT learned tactical reaction or contact GT."""
    def __init__(self, train_pairs, train_metadata):
        self.train=np.asarray(train_pairs)
        self.metadata=train_metadata
        self.features=np.stack([actor_descriptor(x[:,0]) for x in train_pairs])

    def propose(self, actor):
        feature=actor_descriptor(actor)
        dist=((self.features-feature)**2).mean(-1)
        index=int(np.argmin(dist))
        example=self.train[index]
        candidates=[]
        for ja in LIMBS:
            for jb in TARGETS:
                gap=np.linalg.norm(example[:,0,ja]-example[:,1,jb],axis=-1)-RADII[ja]-RADII[jb]
                # Only short contiguous near-surface segments, not penetrating ones.
                mask=(gap>=-.02)&(gap<=.03)
                transitions=np.diff(np.r_[False,mask,False].astype(int))
                for start,end in zip(np.flatnonzero(transitions==1),np.flatnonzero(transitions==-1)):
                    if end-start>=2:
                        candidates.append((end-start,start,end-1,ja,jb))
        info=dict(training_source=self.metadata[index]['source'],training_window=index,
                  actor_descriptor_mse=float(dist[index]),test_reactor_used=False,
                  semantics='geometric near-contact only; hit/block labels not inferred')
        if not candidates:
            return None,info
        _,start,end,ja,jb=max(candidates)
        event=ContactEvent(kind='hit',start=int(start),end=int(end),a_joint=int(ja),b_joint=int(jb),
            provenance='training-only nearest-actor geometric touch proposal; hit is API constraint type, not semantic annotation')
        return event,info


def metrics(actor,pred,truth,fps=20):
    error=pred-truth
    vel=np.diff(pred,axis=0)*fps
    gtvel=np.diff(truth,axis=0)*fps
    bone=lambda x:np.linalg.norm(x[:,1:]-x[:,np.array(PARENTS[1:])],axis=-1)
    bp,bt=bone(pred),bone(truth)
    p=actor[:,LIMBS,None,:]-pred[:,None,TARGETS,:]
    t=actor[:,LIMBS,None,:]-truth[:,None,TARGETS,:]
    radius=RADII[LIMBS,None]+RADII[None,TARGETS]
    pg=np.linalg.norm(p,axis=-1)-radius
    tg=np.linalg.norm(t,axis=-1)-radius
    pp=(pg>=-.02)&(pg<=.03);tp=(tg>=-.02)&(tg<=.03)
    return dict(mpjpe_mm=float(np.linalg.norm(error,axis=-1).mean()*1000),
        velocity_error_mm_s=float(np.linalg.norm(vel-gtvel,axis=-1).mean()*1000),
        acceleration_rms_m_s2=float(np.sqrt(np.mean((np.diff(vel,axis=0)*fps)**2))),
        bone_length_temporal_std_mm=float(bp.std(0).mean()*1000),
        bone_length_error_mm=float(np.abs(bp-bt).mean()*1000),
        proximity_true_positive=int((pp&tp).sum()),proximity_false_positive=int((pp&~tp).sum()),
        proximity_false_negative=int((~pp&tp).sum()),
        proximity_gap_mae_mm=float(np.abs(pg-tg).mean()*1000),
        endpoint_sphere_penetration_fraction=float((pg<-.02).mean()))
