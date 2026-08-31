"""Self-cancellation where the net outputs a DETERMINISTIC answer = the sample itself (one scalar a_t),
scored by the energy score. With a single deterministic output the repulsion term |a-a'| vanishes, so
the energy score reduces to MAE: loss = |a_t - e_{t+1}|. The action a_t pools into the observation
x_{t+1} = e_{t+1} + g*a_t, and the net only ever sees x's. So to predict e_{t+1} well it must RECOMPUTE
its own (deterministic) past action and subtract it -- a pure self-model / self-cancellation test, with
zero apparent entropy by construction and no random bits.

Floor if it cancels perfectly: a_t -> best estimate of e_{t+1} = e_t (random-walk median), MAE ->
E|e_{t+1}-e_t| = sigma_eta*sqrt(2/pi). If it can't cancel its own action, MAE is inflated."""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from selfcancel import RWForecastNet
BASE = os.path.dirname(os.path.abspath(__file__))


def rollout(net, B, T, dev, g, sigma_eta, sigma0=1.0):
    e = torch.randn(B, device=dev) * sigma0
    obs = torch.zeros(B, 0, device=dev)
    mae, acts, es = [], [], []
    for t in range(T):
        a, _ = net(obs)                                          # deterministic scalar output = the action/sample
        e = e + torch.randn(B, device=dev) * sigma_eta
        x = e + g * a
        mae.append((a - e).abs()); acts.append(a); es.append(e)
        obs = torch.cat([obs, x[:, None]], 1)
    st = lambda L: torch.stack(L, 1)
    return st(mae), st(acts), st(es)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--sigma_eta", type=float, default=0.5); ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--d_model", type=int, default=128); ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4); ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eval_every", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(args.seed)
    net = RWForecastNet(args.d_model, args.n_layer, args.n_head, args.T).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    floor = args.sigma_eta * np.sqrt(2 / np.pi)                  # MAE if it cancels perfectly & tracks
    print(f"SELF-CANCEL DETERMINISTIC (energy->MAE) g={args.g} sigma_eta={args.sigma_eta} {args.n_layer}L/d{args.d_model} "
          f"| perfect-cancel MAE floor = {floor:.3f}", flush=True)
    half = args.T // 2; log = []; t0 = time.time()
    for step in range(1, args.steps + 1):
        net.train()
        mae, acts, es = rollout(net, args.batch, args.T, dev, args.g, args.sigma_eta)
        loss = mae.mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % args.eval_every == 0 or step == 1:
            net.eval()
            with torch.no_grad():
                mae, acts, es = rollout(net, 2048, args.T, dev, args.g, args.sigma_eta)
                mae_ss = float(mae[:, half:].mean())
                rec = dict(step=step, mae=float(mae.mean()), mae_ss=mae_ss, floor=floor,
                           mae_vs_floor=mae_ss / floor, mae_by_round=[round(x, 4) for x in mae.mean(0).tolist()])
            log.append(rec)
            print(f"step {step:5d} | MAE {rec['mae']:.3f} | MAE_ss {mae_ss:.3f} (floor {floor:.3f}) "
                  f"ratio {rec['mae_vs_floor']:.3f} "
                  f"({'CANCELS (perceives e)' if rec['mae_vs_floor']<1.3 else 'contaminated'})", flush=True)
    out = os.path.join(BASE, args.out); os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"state": net.state_dict(), "args": vars(args)}, out + ".pt")
    json.dump(dict(args=vars(args), log=log), open(out + ".json", "w"), indent=2)
    print(f"done {time.time()-t0:.0f}s -> {out}.pt | MAE_ss {log[-1]['mae_ss']:.3f} ratio {log[-1]['mae_vs_floor']:.3f}", flush=True)


if __name__ == "__main__":
    main()
