"""v6 confirmatory: (1) scaled internal patch (rule out min-norm weakness) at L1.resid_mid;
(2) clean rubber-hand via obs-desync at a SINGLE step, reading belief shift at that step toward
REAL vs FAKE. The desync is the faithful operationalization given the net re-derives a_{t-1} from obs.
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
import probes, whitebox
import agent as A
BASE=os.path.dirname(os.path.abspath(__file__)); Q=3

def load(tag="d2"):
    ck=torch.load(BASE+f"/runs/agent_{tag}.pt",map_location="cpu"); a=ck["args"]
    A.set_env(a.get("emit",0.6),a.get("stay",0.6))
    net=A.Agent(a["d_model"],a["n_layer"],a["n_head"],a["L"]); net.load_state_dict(ck["state"]); net.eval()
    Wnp={k[9:]:v.numpy().astype(np.float64) for k,v in ck["state"].items() if k.startswith("backbone.")}
    head={k:ck["state"][k].numpy().astype(np.float64) for k in ["act_head.weight","act_head.bias","obs_head.weight","obs_head.bias"]}
    return net,a,Wnp,head
def alog(f,h): return f@h["act_head.weight"].T+h["act_head.bias"]
def sm(z): z=z-z.max(-1,keepdims=True); e=np.exp(z); return e/e.sum(-1,keepdims=True)

def main(tag="d2"):
    net,a,Wnp,head=load(tag)
    L,d,nl,nh,ol=a["L"],a["d_model"],a["n_layer"],a["n_head"],a.get("open_loop",False)
    T=torch.tensor(A.T0,dtype=torch.float32);E=torch.tensor(A.EM,dtype=torch.float32);pi=torch.tensor(A.PI,dtype=torch.float32)
    with torch.no_grad(): obs,st=A.rollout(net,600,L,"cpu",T,E,pi,a.get("det_action",False),ol)
    obs=obs.numpy(); st=st.numpy()
    acts0,_=whitebox.forward(Wnp,obs,nl,nh)
    p0=sm(alog(acts0["final_ln"],head)); net_a=p0.argmax(-1)
    pa=np.zeros_like(net_a); pa[:,1:]=net_a[:,:-1]; tail=slice(L//2,None)
    bel0=np.stack([A.oracle_filter(obs[i],p0[i],ol) for i in range(len(obs))])
    Xr=acts0["L1.resid_post"].reshape(-1,d); Wb,bb,_=probes.ridge_fit(Xr,bel0.reshape(-1,Q))
    print(f"=== v6  agent_{tag} ===")

    # (1) scaled internal patch at L1.resid_mid, a_{t-1}->a+1, scale up to overdrive
    PATCH="L1.resid_mid"; Xp=acts0[PATCH].reshape(-1,d)
    Wa,ba,_=probes.ridge_fit(Xp,np.eye(Q)[pa.reshape(-1)]); pinv=np.linalg.solve(Wa.T@Wa+1e-6*np.eye(Q),Wa.T)
    Wro,bro,_=probes.ridge_fit(Xr,np.eye(Q)[pa.reshape(-1)])
    print("  (1) overdriven internal patch a_(t-1)->a+1 @L1.resid_mid:")
    for g in (1,3,8,20):
        tgt=(pa+1)%Q
        def fn(name,x,g=g,tgt=tgt):
            if name!=PATCH: return x
            x=x.copy(); cur=x.reshape(-1,d)@Wa+ba
            return x+(g*(np.eye(Q)[tgt.reshape(-1)]-cur)@pinv).reshape(x.shape)
        actsP,_=whitebox.forward(Wnp,obs,nl,nh,edit_fn=fn)
        XrP=actsP["L1.resid_post"].reshape(-1,d)
        stick=((XrP@Wro+bro).argmax(1)==tgt.reshape(-1)).mean()
        belP=probes.readout(XrP,Wb,bb).reshape(len(obs),L,Q)
        belF=np.stack([A.oracle_filter(obs[i],np.eye(Q)[(pa[i]+1)%Q],ol) for i in range(len(obs))])
        af=(belP.argmax(-1)[:,tail]==belF.argmax(-1)[:,tail]).mean()
        ar=(belP.argmax(-1)[:,tail]==bel0.argmax(-1)[:,tail]).mean()
        print(f"     g={g:>2}: patch sticks(post)={stick:.3f} | belief-MAP agrees REAL={ar:.3f} FAKE={af:.3f}")

    # (2) clean rubber-hand: single-step obs desync at fixed t0, read belief at t0 toward REAL/FAKE.
    # FAKE here = the net should infer state from emission e=(o_desync - own_a). Build oracle on
    # the desynced obs for the position(s) >= t0.
    print("\n  (2) obs-desync rubber-hand (faithful): shift obs by +delta from t0 on; belief toward REAL vs FAKE")
    t0=L//2
    for delta in (1,2):
        obs2=obs.copy(); obs2[:,t0:]=(obs2[:,t0:]+delta)%Q
        acts2,_=whitebox.forward(Wnp,obs2,nl,nh)
        p2=sm(alog(acts2["final_ln"],head)); m2=p2.argmax(-1)
        X2=acts2["L1.resid_post"].reshape(-1,d); bel2p=probes.readout(X2,Wb,bb).reshape(len(obs),L,Q)
        # REAL hypothesis: belief had obs NOT been desynced (clean). FAKE: oracle on desynced obs.
        belF=np.stack([A.oracle_filter(obs2[i],p2[i],ol) for i in range(len(obs))])
        sl=slice(t0,None)
        ar=(bel2p.argmax(-1)[:,sl]==bel0.argmax(-1)[:,sl]).mean()   # vs clean (REAL world)
        af=(bel2p.argmax(-1)[:,sl]==belF.argmax(-1)[:,sl]).mean()   # vs desynced oracle (FAKE)
        nf=(m2[:,sl]==belF.argmax(-1)[:,sl]).mean()
        print(f"     delta=+{delta}: net belief-MAP agrees clean-REAL={ar:.3f}  desynced-FAKE={af:.3f} | net-action==FAKE-MAP={nf:.3f}")

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "d2")
