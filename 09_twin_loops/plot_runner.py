"""Figure for runner_world.json: persistence Delta vs horizon, per coordinate and baseline."""
import json
import matplotlib, matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = json.load(open('results/runner_world.json'))
H = [2, 8, 20, 50]
names = [('persona_model', 'actor persona\n(models rain)'),
         ('persona_outcome', 'actor persona\n(identifies with outcomes)'),
         ('reader', 'reader of the actor'),
         ('temperature', 'temperature\n(consequential)')]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharex=True)
for ax, (k, title) in zip(axes, names):
    for (b, row), c in zip(r[k].items(), ('C0', 'C1', 'C2')):
        lab = b.replace('baseline_', 'baseline q ∈ [').replace('-', ', ') + ']'
        ax.plot(H, [row[f'Delta_{h}'] for h in H], 'o-', color=c, label=lab)
    ax.axhline(1, color='gray', ls=':', lw=.8); ax.axhline(0, color='gray', ls=':', lw=.8)
    ax.set_xscale('log'); ax.set_xticks(H); ax.set_xticklabels([str(h) for h in H])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_title(title, fontsize=10); ax.set_xlabel('weeks after perturbation')
    if k == 'persona_outcome': ax.set_ylim(-0.2, 4.6)
    else: ax.set_ylim(-0.2, 1.3)
axes[0].set_ylabel('persistence Δ\n(1 = kept, 0 = reverted)')
axes[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig('figs/runner_persistence.png', dpi=140)
print('saved')
