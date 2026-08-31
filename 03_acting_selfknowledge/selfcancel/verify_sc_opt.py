import numpy as np
# Self-cancellation steady-state optimum.
# State: P=posterior var of latent; S=P+q predictive var (q=sigma_eta^2). Control: R=(g*sigma)^2.
# Dynamics (Kalman, random walk): P' = S*R/(S+R), S'=P'+q  =>  steady S(R)=(q+sqrt(q^2+4qR))/2.
# Per-step avg reward (steady state): rew(R) = -0.5*( S(R)/sigma^2 + log(2*pi*sigma^2) ), sigma^2=R/g^2.
def closed_ratio2(g):  # sigma_opt^2 / S_opt   (over-confidence ratio^2); INDEPENDENT of q
    return ((g**2-1)+np.sqrt(1+6*g**2+g**4))/(4*g**2)
def brute(g,q,Rs=None):
    Rs=np.logspace(-6,3,200000) if Rs is None else Rs
    S=(q+np.sqrt(q**2+4*q*Rs))/2; sig2=Rs/g**2
    rew=-0.5*(S/sig2+np.log(2*np.pi*sig2))
    i=rew.argmax(); R=Rs[i]; S0=(q+np.sqrt(q**2+4*q*R))/2; s2=R/g**2
    return np.sqrt(s2/S0), np.sqrt(s2), np.sqrt(S0)   # ratio, sigma_opt, sigma_honest
def fhdp(g,q,T=40,Rs=None):  # finite-horizon backward DP, report mid-episode (steady) sigma
    Rs=np.logspace(-5,2.5,4000) if Rs is None else Rs
    # value over P (grid), backward; track optimal sigma per (t,P). Report sigma at t=T//2 from the prior-driven P-path.
    Pgrid=np.logspace(-4,1.5,400)
    V=np.zeros((T+1,len(Pgrid)))
    polR=np.zeros((T,len(Pgrid)))
    for t in range(T-1,-1,-1):
        for j,P in enumerate(Pgrid):
            S=P+q; sig2=Rs/g**2
            r=-0.5*(S/sig2+np.log(2*np.pi*sig2))
            Pn=S*Rs/(S+Rs)
            Vn=np.interp(Pn,Pgrid,V[t+1])
            tot=r+Vn; k=tot.argmax(); V[t,j]=tot[k]; polR[t,j]=Rs[k]
    # roll the P-path forward under the policy
    P=1.0
    for t in range(T):
        R=np.interp(P,Pgrid,polR[t]); S=P+q; 
        if t==T//2: smid=np.sqrt(R/g**2); hmid=np.sqrt(S)
        P=S*R/(S+R)
    return smid/hmid, smid, hmid
print(f"{'g':>4} | {'closed ratio':>12} | {'brute ratio':>11} | {'DP(T40) ratio':>13} | sigma_opt/honest(brute)")
for g in [0.0001,0.5,1.0,2.0,5.0,100.0]:
    cr=np.sqrt(closed_ratio2(g)); br,so,sh=brute(g,0.25)
    dp=fhdp(g,0.25) if g>0.01 else (1.0,0,0)
    print(f"{g:>4} | {cr:>12.4f} | {br:>11.4f} | {dp[0]:>13.4f} | {so:.4f}/{sh:.4f}")
print("\nasymptote g->inf: ratio -> 1/sqrt(2) =", 1/np.sqrt(2))
