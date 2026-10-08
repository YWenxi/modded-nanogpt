import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse(path):
    train_steps, train_loss = [], []
    val_steps, val_loss = [], []
    pat_train = re.compile(r"step:(\d+)/\d+ train_loss:([\d.]+)")
    pat_val = re.compile(r"step:(\d+)/\d+ val_loss:([\d.]+)")
    with open(path) as f:
        for line in f:
            m = pat_train.search(line)
            if m:
                train_steps.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
            m = pat_val.search(line)
            if m:
                val_steps.append(int(m.group(1)))
                val_loss.append(float(m.group(2)))
    return (train_steps, train_loss), (val_steps, val_loss)

def smooth(xs, w=25):
    out, acc = [], 0.0
    for i, x in enumerate(xs):
        acc += x
        if i >= w:
            acc -= xs[i - w]
        out.append(acc / min(i + 1, w))
    return out

runs = [
    ("logs/compare_ns.log", "Newton-Schulz", "tab:blue"),
    ("logs/compare_polar_sf.log", "Polar Express (safety 1.01)", "tab:orange"),
]
data = [(parse(p), label, color) for p, label, color in runs]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for (ts, tl), (vs, vl), label, color in [(d[0][0], d[0][1], d[1], d[2]) for d in data]:
    ax1.plot(ts, tl, color=color, alpha=0.12, linewidth=0.8)
for (ts, tl), (vs, vl), label, color in [(d[0][0], d[0][1], d[1], d[2]) for d in data]:
    ax1.plot(ts, smooth(tl), color=color, label=f"{label} train", linewidth=1.8)
    ax1.plot(vs, vl, color=color, linestyle="--", marker="o", markersize=4,
             label=f"{label} val", linewidth=1.4)
ax1.set_xlabel("step"); ax1.set_ylabel("loss")
ax1.set_title("Full run")
ax1.legend(); ax1.grid(alpha=0.3)

# zoomed tail: val loss + smoothed train loss, steps >= 1000
for (ts, tl), (vs, vl), label, color in [(d[0][0], d[0][1], d[1], d[2]) for d in data]:
    mask = [i for i, s in enumerate(ts) if s >= 1000]
    ax2.plot([ts[i] for i in mask], [smooth(tl)[i] for i in mask],
             color=color, label=f"{label} train (smoothed)", linewidth=1.8)
    vmask = [i for i, s in enumerate(vs) if s >= 1000]
    ax2.plot([vs[i] for i in vmask], [vl[i] for i in vmask], color=color,
             linestyle="--", marker="o", markersize=5, label=f"{label} val", linewidth=1.4)
ax2.set_xlabel("step"); ax2.set_ylabel("loss")
ax2.set_title("Zoomed: steps 1000+")
ax2.legend(); ax2.grid(alpha=0.3)

fig.suptitle("Muon: Newton-Schulz vs Polar Express orthogonalization (1695 steps, 8xA100)")
fig.tight_layout()
out = "logs/compare_polar_vs_ns.png"
fig.savefig(out, dpi=150)
print(out)

for (ts, tl), (vs, vl), label, _ in [(d[0][0], d[0][1], d[1], d[2]) for d in data]:
    print(f"{label}: final train={tl[-1]:.4f}, final val={vl[-1]:.4f}, "
          f"val trajectory last 5: {[f'{v:.4f}' for v in vl[-5:]]}")
ns_val = data[0][0][1][1]
pe_val = data[1][0][1][1]
print("val diff (PE - NS) per eval:", [f"{p - n:+.4f}" for n, p in zip(ns_val, pe_val)])
