"""Rubber-hand v5: NECESSITY + correct-direction test.

(A) NECESSITY: ablate the a_{t-1} subspace (project it out) at each layer; measure how much the net's
    action accuracy and belief degrade. If the trace is used, ablation hurts.
(B) ATTENTION KNOCKOUT: in block 1, zero the attention from position t to t-1 (the immediate past obs
    that carries a_{t-1} & the freshest emission). Measure belief/action degradation.
(C) The 'rubber-hand done right': feed a DESYNCED observation. At a chosen step the env actually used
    action a, but we hand the net o' = o + (a'-a) for that one step onward-read, i.e. consistent with
    having acted a'. Then check: does the net's inferred state shift like decode-with-(o', own-a)?
    This separates 'subtract my own action' from 'read the obs'. Compared against the internal-patch
    null result it tells us whether the net keys off obs (re-derived) vs a stored efference vector.
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
    with torch.no_grad():
        obs,st=A.rollout(net,600,L,"cpu",T,E,pi,a.get("det_action",False),ol)
    obs=obs.numpy(); st=st.numpy()
    acts0,_=whitebox.forward(Wnp,obs,nl,nh)
    p0=sm(alog(acts0["final_ln"],head)); net_a=p0.argmax(-1)
    pa=np.zeros_like(net_a); pa[:,1:]=net_a[:,:-1]
    tail=slice(L//2,None)
    base_acc=(net_a[:,tail]==st[:,tail]).mean()
    print(f"=== v5 NECESSITY  agent_{tag} ({nl}x{d}) | clean act_acc(tail)={base_acc:.3f} ===")

    # (A) ablate a_{t-1} subspace at each layer (project out the readout column space)
    for layer in ["L0.resid_post","L1.resid_mid","L1.resid_post"]:
        X=acts0[layer].reshape(-1,d)
        Wa,ba,r2=probes.ridge_fit(X,np.eye(Q)[pa.reshape(-1)])
        Qa,_=np.linalg.qr(Wa)                                   # (d,3) orthonormal basis of trace subspace
        def ablate(name,x,Qa=Qa,layer=layer):
            if name!=layer: return x
            xf=x.reshape(-1,d); proj=xf@Qa@Qa.T
            return (xf-proj).reshape(x.shape)
        actsA,_=whitebox.forward(Wnp,obs,nl,nh,edit_fn=ablate)
        pa_acc=(sm(alog(actsA["final_ln"],head)).argmax(-1)[:,tail]==st[:,tail]).mean()
        print(f"  ablate a_(t-1) subspace @ {layer} (R2={r2:.3f}): act_acc(tail) {base_acc:.3f} -> {pa_acc:.3f}  (Δ={pa_acc-base_acc:+.3f})")

    # (B) block-1 attention knockout: zero attention weight from t to t-1 (renormalize). Implement by
    # editing block-1 input so that... simplest: custom forward zeroing the (t,t-1) attn entry.
    # We re-run a minimal mha with masking of the immediate-previous key.
    knock_b1_prev(Wnp,head,obs,st,nl,nh,d,tail,base_acc)

    # (C) desynced observation: at every step t>=L//2, replace o_t with (o_t + delta)%Q (delta=a'-a),
    # which is the observation the env WOULD have produced had the net acted a'=a+1 last step
    # (same emission). The net subtracts ITS OWN committed action; so its decoded emission becomes
    # e + delta. If it keys off obs, its state estimate shifts by +delta vs clean. Compare to oracle
    # run on the desynced obs with the net's own action dist.
    for delta in (1,2):
        obs2=obs.copy(); obs2[:,L//2:]=(obs2[:,L//2:]+delta)%Q
        acts2,_=whitebox.forward(Wnp,obs2,nl,nh)
        p2=sm(alog(acts2["final_ln"],head)); m2=p2.argmax(-1)
        # oracle belief on the DESYNCED obs with net's own (recomputed) action dist
        bel2=np.stack([A.oracle_filter(obs2[i],p2[i],ol) for i in range(len(obs))])
        agr=(m2[:,tail]==bel2.argmax(-1)[:,tail]).mean()
        print(f"  [C] desync obs by +{delta} (tail): net-action follows oracle-on-desynced-obs = {agr:.3f}")


def knock_b1_prev(Wnp,head,obs,st,nl,nh,d,tail,base_acc):
    """Re-implement forward zeroing block-1 attention from each query t to key t-1."""
    def edit(name,x): return x
    # Patch by hooking into mha through a custom run: easiest is to monkey a forward that, in block 1,
    # sets scores[...,t,t-1] = -inf. We reuse whitebox primitives.
    from whitebox import layernorm, gelu, mha
    def fwd_knock(W,tokens):
        L=tokens.shape[-1]
        x=W["tok.weight"][tokens]+W["pos.weight"][np.arange(L)]
        for i in range(nl):
            p=f"blocks.{i}."
            h1=layernorm(x,W[p+"ln1.weight"],W[p+"ln1.bias"])
            if i==1:
                B,Ld,dd=h1.shape; hd=dd//nh
                Wi,bi=W[p+"attn.in_proj_weight"],W[p+"attn.in_proj_bias"]
                qkv=h1@Wi.T+bi; q,k,v=qkv[...,:dd],qkv[...,dd:2*dd],qkv[...,2*dd:]
                sp=lambda t:t.reshape(B,Ld,nh,hd).transpose(0,2,1,3)
                q,k,v=sp(q),sp(k),sp(v)
                sc=q@k.transpose(0,1,3,2)/np.sqrt(hd)
                mask=np.triu(np.ones((Ld,Ld),bool),1)
                sc=np.where(mask[None,None],-np.inf,sc)
                for t in range(1,Ld): sc[:,:,t,t-1]=-np.inf            # knock out t->t-1
                from whitebox import softmax_lastdim
                at=softmax_lastdim(sc); ctx=(at@v).transpose(0,2,1,3).reshape(B,Ld,dd)
                aout=ctx@W[p+"attn.out_proj.weight"].T+W[p+"attn.out_proj.bias"]
            else:
                aout,_,_=mha(h1,W,p+"attn.",nh)
            x=x+aout
            h2=layernorm(x,W[p+"ln2.weight"],W[p+"ln2.bias"])
            m=gelu(h2@W[p+"mlp.0.weight"].T+W[p+"mlp.0.bias"])@W[p+"mlp.2.weight"].T+W[p+"mlp.2.bias"]
            x=x+m
        xf=layernorm(x,W["lnf.weight"],W["lnf.bias"]); return xf
    xf=fwd_knock(Wnp,obs)
    acc=(sm(alog(xf,head)).argmax(-1)[:,tail]==st[:,tail]).mean()
    print(f"  [B] block-1 attn knockout t->t-1: act_acc(tail) {base_acc:.3f} -> {acc:.3f} (Δ={acc-base_acc:+.3f})")


if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "d2")
