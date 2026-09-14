"""Two-state Viterbi actor correspondence; no attack/defense role inference."""
import numpy as np


def stabilize_actor_identity(positions):
    x=np.asarray(positions)
    if x.ndim!=4 or x.shape[1]!=2 or x.shape[-1]!=3 or len(x)<2 or not np.isfinite(x).all():
        raise ValueError('Expected finite [T>=2,2,J,3] shared-frame positions')
    permutations=np.array([[0,1],[1,0]])
    cost=np.full((len(x),2),np.inf);cost[0,0]=0
    back=np.zeros((len(x),2),dtype=np.int64)
    for t in range(1,len(x)):
        for state in range(2):
            transition=np.array([np.mean((x[t,permutations[state]]-x[t-1,permutations[previous]])**2) for previous in range(2)])
            candidates=cost[t-1]+transition
            best=int(np.argmin(candidates));back[t,state]=best;cost[t,state]=candidates[best]
    states=np.zeros(len(x),dtype=np.int64);states[-1]=int(np.argmin(cost[-1]))
    for t in range(len(x)-1,0,-1):states[t-1]=back[t,states[t]]
    indices=permutations[states]
    fixed=x[np.arange(len(x))[:,None],indices]
    return fixed,indices,{'frames_permuted':int(states.sum()),'identity_switches_corrected':int(np.count_nonzero(np.diff(states))),
        'raw_continuity_cost':float(np.sum(np.mean(np.diff(x,axis=0)**2,axis=(1,2,3)))),
        'corrected_continuity_cost':float(cost[-1].min()),
        'semantics':'geometric actor correspondence anchored to frame0; not causal or attack-role labels',
        'limitation':'Close overlap can be ambiguous. This is deterministic engineering postprocessing, not physics validation.'}
