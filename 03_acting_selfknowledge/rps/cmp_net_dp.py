import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/comp_icl")); torch.set_num_threads(8)
from rps_im import RPSNet, rollout
DEV="cpu"; rng=np.random.default_rng(0)
dpr=json.load(open("dp_results.json"))["results"]
@torch.no_grad()
def evalnet(beta, T, B=6000):
    ck=torch.load(f"rps_runs/rpstraj_b{beta}.pt",map_location=DEV); a=ck["args"]
    net=RPSNet(a["d_model"],a["n_layer"],a["n_head"],a["T"]); net.load_state_dict(ck["state"]); net.eval()
    is_br=torch.rand(B)<beta
    g=rng.gamma(0.5,1.0,size=(B,3)); bias=torch.tensor(g/g.sum(1,keepdims=True),dtype=torch.float32)
    _,_,pay,ent=rollout(net,B,T,DEV,beta,bias,per_traj=True,is_br=is_br)
    pg=pay.mean(1).numpy(); eg=ent.mean(1).numpy(); m=(~is_br).numpy()
    return dict(pay=pg.mean(), pay_bias=pg[m].mean(), pay_br=pg[~m].mean(),
                H=eg.mean(), H_bias=eg[m].mean(), H_br=eg[~m].mean(), train_T=a["T"])
print("NET (per-traj) vs DP-optimal, matched horizon T=30   [trained T=40]")
print("beta |  net_pay  net_bias  net_BR | DP_opt  DP_bias  DP_BR | net_H_bias  DP_H_bias | gap(bias)")
for b in [0.2,0.3,0.4,0.5]:
    n=evalnet(b,30); d=dpr[str(b)]
    dHb=d.get("meanH_bias_games",float("nan"))
    print(f"{b}  | {n['pay']:+.3f}   {n['pay_bias']:+.3f}   {n['pay_br']:+.3f} | "
          f"{d['payoff_opt']:+.3f}  {d['payoff_bias_games']:+.3f}  {d['payoff_br_games']:+.3f} | "
          f"   {n['H_bias']:.2f}       {dHb:.2f}   | {d['payoff_bias_games']-n['pay_bias']:+.3f}")
print("\nNET at native T=40 (sanity that the gap isn't a horizon artifact):")
for b in [0.2,0.3,0.4,0.5]:
    n=evalnet(b,40)
    print(f"  beta={b}: net_pay={n['pay']:+.3f} bias={n['pay_bias']:+.3f} BR={n['pay_br']:+.3f} H_bias={n['H_bias']:.2f}")
