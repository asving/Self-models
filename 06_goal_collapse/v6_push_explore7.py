"""O1 certification: SHARED-ROAD orchard (S=9, ADJACENT tops).
(a) opacity funnel: Bayes 2-hyp observer posterior-on-truth vs rounds-into-
segment, committed play — want flat ~0.5 until endgame (vs S=5 baseline);
(b) deferral gap: coherence reward of committed vs fork-deferrer — expect ~0
(deferral free) => the arm tests SPONTANEOUS binding, not forced."""
import numpy as np

def run(S, adjacent, policy, N=6000, T=64, seed=0, rho=0.3,
        sigma=.8, alpha=.75, c=.35, k=4):
    rng = np.random.default_rng(seed)
    RD = np.array([[min(abs((g-s)%S), S-abs((g-s)%S)) for g in range(S)]
                   for s in range(S)])
    DST = np.zeros((S,S),int)
    for s in range(S):
        for g in range(S):
            diff = (g-s)%S
            DST[s,g] = 0 if diff==0 else (1 if diff<=S//2 else 2)
    DV = np.array([0,1,-1])
    s = rng.integers(0,S,N)
    t1 = rng.integers(0,S,N)
    t2 = (t1+1)%S if adjacent else (t1+1+rng.integers(0,S-1,N))%S
    tlo, thi = np.minimum(t1,t2), np.maximum(t1,t2)
    b = np.full((N,S),1/S)
    run_=np.zeros(N,int); eprev=np.full(N,-1)
    g_side = (rng.random(N)<.5).astype(int)     # committed side 0=lo,1=hi
    picked = np.zeros(N,bool) if policy=='defer' else np.ones(N,bool)
    cnt=np.zeros((N,2)); coh=np.zeros(N); dseg=np.ones(N,int)
    post=np.full((N,2),.5)
    fun_n=np.zeros(80); fun_p=np.zeros(80)
    rows=np.arange(N)
    Tb=np.full((S,S),(1-sigma)/2*0); 
    for i in range(S):
        Tb[i,i]=sigma; Tb[i,(i+1)%S]+= (1-sigma)/2; Tb[i,(i-1)%S]+=(1-sigma)/2
    L=np.full((S,S),(1-alpha)/(S-1)); np.fill_diagonal(L,alpha)
    Pd=np.zeros((3,S,S))
    for i in range(S):
        Pd[0,i,i]=1; Pd[1,i,(i+1)%S]=1; Pd[2,i,(i-1)%S]=1
    ntop=np.zeros(N)
    for t in range(T):
        sh = b.argmax(1)
        tt = np.where(g_side==0, tlo, thi)
        if policy=='defer':
            near = np.minimum(RD[sh,tlo], RD[sh,thi]) <= 1
            newly = near & ~picked
            if newly.any():
                idx=np.where(newly)[0]
                g_side[idx] = (rng.random(len(idx))<.5).astype(int)
                picked[idx]=True
            tgt_far = np.where(RD[sh,tlo]<=RD[sh,thi], tlo, thi)
            tt = np.where(picked, np.where(g_side==0,tlo,thi), tgt_far)
        a_star = DST[sh, tt]
        a = np.where(rng.random(N)<rho, rng.integers(0,3,N), a_star)
        # observer update (committed-model likelihood, both sides)
        for side,tg in ((0,tlo),(1,thi)):
            match = a == DST[sh,tg]
            post[:,side] *= (1-rho)*match + rho/3
        post /= post.sum(1,keepdims=True)
        m = dseg<80
        np.add.at(fun_n, dseg[m], 1)
        np.add.at(fun_p, dseg[m], post[rows,g_side][m])
        cnt[:,0]+= a==DST[sh,tlo]; cnt[:,1]+= a==DST[sh,thi]
        # world
        push = rng.random(N)<c
        sp=(s+DV[a])%S
        stay=rng.random(N)<sigma
        hop=1-2*(rng.random(N)<.5).astype(int)
        s=np.where(push,sp,np.where(stay,s,(s+hop)%S)).astype(int)
        x=np.where(rng.random(N)<alpha,s,(s+1+rng.integers(0,S-1,N))%S).astype(int)
        Tn=(1-c)*Tb[None]+c*Pd[a]
        b=np.einsum('ni,nij->nj',b,Tn)*L[:,x].T
        b/=b.sum(1,keepdims=True)
        run_=np.where(x==eprev,run_+1,1); eprev=x
        hm=run_>=k
        istop=hm&((x==tlo)|(x==thi))
        dseg+=1
        if istop.any():
            idx=np.where(istop)[0]
            ntop[idx]+=1
            coh[idx]+=cnt[idx].max(1); cnt[idx]=0
            surv=np.where(x[idx]==tlo[idx], thi[idx], tlo[idx])
            if adjacent:
                new=(surv+(surv-x[idx]))%S
            else:
                new=np.zeros(len(idx),int)
                for j,(lo,hi) in enumerate(zip(tlo[idx],thi[idx])):
                    junk=[q for q in range(S) if q!=lo and q!=hi]
                    new[j]=junk[rng.integers(0,len(junk))]
            tlo[idx]=np.minimum(surv,new); thi[idx]=np.maximum(surv,new)
            g_side[idx]=(rng.random(len(idx))<.5).astype(int)
            picked[idx] = policy!='defer'
            post[idx]=.5; dseg[idx]=1
        run_[hm]=0
    coh+=cnt.max(1)
    return 4*coh.mean()/T, ntop.mean(), fun_p/np.maximum(fun_n,1), fun_n

print('S=5 baseline (non-adjacent tops), committed:')
r,nt,fp,fn = run(5, False, 'commit')
print(f'  R_coh={r:.3f} tops={nt:.2f}  funnel d=1..8:',
      ' '.join(f'{fp[d]:.2f}' for d in range(1,9)))
print('S=9 SHARED ROAD (adjacent tops), committed:')
r,nt,fp,fn = run(9, True, 'commit')
print(f'  R_coh={r:.3f} tops={nt:.2f}  funnel d=1..16:',
      ' '.join(f'{fp[d]:.2f}' for d in range(1,17)))
print('S=9 SHARED ROAD, fork-deferrer:')
r2,nt2,fp2,_ = run(9, True, 'defer')
print(f'  R_coh={r2:.3f} tops={nt2:.2f}  funnel d=1..16:',
      ' '.join(f'{fp2[d]:.2f}' for d in range(1,17)))
print(f'DEFERRAL GAP (committed - deferrer): {r-r2:+.3f}')
