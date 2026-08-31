"""THREAD 5 (efficient): training evolution of the tie-choice.

Batches all tie-context policy queries per checkpoint (group by sequence length).
Tracks corner/center/edge choice fractions, the postype-CONTROLLED legibility
preference (clean self-legibility signal), and opening-center prob, vs the training
iteration / play-strength / tie-entropy.
"""
import collections, os, sys
import numpy as np
import torch, torch.nn.functional as F
import ttt
from model import TTTNet
import policy_eval as PE
from ttt_collapse_search import (
    enumerate_contexts, build_legibility, child_legibility_of_move, RUNS)

CENTER={4}; CORNERS={0,2,6,8}
def postype(c): return "center" if c in CENTER else ("corner" if c in CORNERS else "edge")
torch.set_num_threads(1)


def prep():
    contexts = enumerate_contexts(p_strong=0.5)
    key_H = build_legibility(contexts)
    ties = {c["true_seq"]: c for c in contexts if len(c["opt"]) > 1}
    uties = list(ties.values())
    for c in uties:
        c["move_leg"]={nm:child_legibility_of_move(list(c["board"]),nm,c["occ_seq"],c["round"],key_H) for nm in c["opt"]}
        v=np.array(list(c["move_leg"].values())); c["leg_spread"]=float(v.max()-v.min())
    disc=[c for c in uties if c["leg_spread"]>1e-4]
    ctl=[]
    for c in disc:
        opt=list(c["opt"]); legs=np.array([c["move_leg"][m] for m in opt])
        hi,lo=int(legs.argmax()),int(legs.argmin())
        if postype(opt[hi])==postype(opt[lo]): ctl.append((c,opt[hi],opt[lo]))
    c0=[c for c in contexts if c["round"]==0][0]
    return uties, ctl, c0


@torch.no_grad()
def batch_policy(model, fullobs, ctxs):
    """ctxs: list of context dicts. Returns list of full 9-dim legal-renorm policy."""
    by_len=collections.defaultdict(list)
    for i,c in enumerate(ctxs): by_len[len(c["true_seq"])].append(i)
    out=[None]*len(ctxs)
    for L,idxs in by_len.items():
        seqs=np.stack([[PE.obs_from_board(list(b),fullobs) for b in ctxs[i]["true_seq"]] for i in idxs])
        inp=torch.tensor(seqs,dtype=torch.float32)
        logits=model(inp)[:,-1]
        for k,i in enumerate(idxs):
            board=ctxs[i]["board"]
            lg=logits[k].clone()
            legal=torch.tensor([0.0 if cc!=0 else 1.0 for cc in board])
            lg=lg.masked_fill(legal==0,-1e9)
            out[i]=F.softmax(lg,-1).numpy()
    return out


def metrics(model, fullobs, uties, ctl, c0):
    P=batch_policy(model,fullobs,uties)
    cn=collections.Counter()
    for c,p in zip(uties,P):
        opt=list(c["opt"]); mv=opt[int(np.array([p[m] for m in opt]).argmax())]; cn[postype(mv)]+=1
    n=len(uties)
    ctl_ctxs=[c for c,_,_ in ctl]
    Pc=batch_policy(model,fullobs,ctl_ctxs)
    pref=[]
    for (c,hi,lo),p in zip(ctl,Pc):
        s=p[hi]+p[lo]
        if s>0: pref.append(p[hi]/s)
    po=batch_policy(model,fullobs,[c0])[0]
    return dict(corner=cn["corner"]/n,center=cn["center"]/n,edge=cn["edge"]/n,
                leg_pref_ctl=float(np.mean(pref)),open_center=float(po[4]))


def load_step(path):
    ck=torch.load(path,map_location="cpu",weights_only=False)
    m=TTTNet(**ck["config"]); m.load_state_dict(ck["state_dict"]); m.eval()
    return m, bool(ck.get("fullobs",False)), ck.get("it",None), ck


def main():
    uties,ctl,c0=prep()
    print(f"ties={len(uties)} discriminating-controlled={len(ctl)}",flush=True)
    for tag in ["rl","rl_fullobs","onpolicy_teacher"]:
        sd=f"{RUNS}/{tag}_steps"
        if not os.path.isdir(sd): continue
        print(f"\n=== {tag} ===",flush=True)
        print(f"{'it':>6} {'corner':>7} {'center':>7} {'edge':>7} {'legprefC':>9} {'openCtr':>8} {'vsRandW':>8} {'tieH':>7}",flush=True)
        for s in sorted(os.listdir(sd)):
            m,fo,it,ck=load_step(os.path.join(sd,s))
            d=metrics(m,fo,uties,ctl,c0)
            vr=ck.get("vs_random",{}); w=vr.get("win",float('nan')) if isinstance(vr,dict) else float('nan')
            tieH=ck.get("collapse",{}).get("tie_entropy",float('nan'))
            print(f"{str(it):>6} {d['corner']:7.3f} {d['center']:7.3f} {d['edge']:7.3f} {d['leg_pref_ctl']:9.3f} {d['open_center']:8.3f} {w:8.3f} {tieH:7.3f}",flush=True)


if __name__=="__main__":
    main()
