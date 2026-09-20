cat > /mnt/user-data/outputs/loss_logger_patch.py << 'ENDOFFILE'
"""
Paste this into your train.py in the appropriate locations.
Marked clearly with # ── ADD THIS ── comments.
"""

# ── ADD THIS: at the top of main(), after log_dir.mkdir() ──────────────────
import csv
from pathlib import Path

LOG_CSV = log_dir / "training_losses.csv"

# Write header only if starting fresh or file doesn't exist
if not LOG_CSV.exists() or args.fresh:
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "wm", "kl", "recon", "rew",
            "actor", "crit", "nan_skips",
            "ep_ret", "mean20", "grasp_pct",
            "vram_gb", "timestamp"
        ])
    print(f"  Loss log: {LOG_CSV}")


# ── ADD THIS: replace your existing log_every block in the main loop ────────
# Find this existing block in your train.py:
#
#   if total_steps % cfg["training"]["log_every"] == 0 and losses:
#       vram = ...
#       print(f"  loss wm=...")
#
# Replace it with:

if total_steps % cfg["training"]["log_every"] == 0 and losses:
    vram = torch.cuda.memory_allocated(0)/1e9 if device.type=="cuda" else 0
    gr   = buf.grasp_rate() if hasattr(buf, 'grasp_rate') else 0.0
    mean20 = np.mean(list(ep_ret_history)[-20:]) if ep_ret_history else 0

    # Print to console as before
    print(f"  loss wm={losses['wm']:.3f}  kl={losses['kl']:.3f}  "
          f"recon={losses['recon']:.4f}  rew={losses['rew']:.4f}  "
          f"actor={losses['actor']:.3f}  crit={losses['crit']:.3f}  "
          f"nan={nan_skips}  vram={vram:.1f}GB  "
          f"grasp%={gr*100:.1f}")

    # Write to CSV log
    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            total_steps,
            round(losses['wm'],    4),
            round(losses['kl'],    4),
            round(losses['recon'], 5),
            round(losses['rew'],   6),
            round(losses['actor'], 4),
            round(losses['crit'],  5),
            nan_skips,
            round(ep_ret_history[-1] if ep_ret_history else 0, 3),
            round(mean20, 3),
            round(gr * 100, 1),
            round(vram, 2),
            datetime.now().strftime("%H:%M:%S")
        ])
ENDOFFILE
echo "patch written"