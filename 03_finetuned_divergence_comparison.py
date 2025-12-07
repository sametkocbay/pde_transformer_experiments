import torch
import numpy as np
from pathlib import Path
from pdetransformer.core.mixed_channels import PDETransformer
from src.data_loader import find_first_h5, load_shear_velocity
from src.divergence_utils import compute_divergence, divergence_stats
import matplotlib.pyplot as plt

# Gerät
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Test-Datensätze aus data_splits.txt laden (gleicher Split wie 00/01)
splits_file = Path('checkpoints') / 'data_splits.txt'
test_dirs = []
if splits_file.exists():
    with open(splits_file, 'r') as f:
        lines = f.readlines()
    in_test = False
    for line in lines:
        if line.strip() == "TEST:":
            in_test = True
            continue
        if in_test and line.strip().startswith("sf_"):
            test_dirs.append(Path("dataset") / line.strip())
print(f"Test-Datensätze gefunden: {len(test_dirs)}")

# Ersten Test-Datensatz verwenden
dataset_dir = str(test_dirs[0])
filename = str(find_first_h5(dataset_dir))
print(f"Evaluiere auf: {dataset_dir}")

# Lade Daten
d = load_shear_velocity(filename)
vel = d['vel']
time = d['time']
x = d['x']
z = d['z']

# Divergenz-finetuned Modell laden
print("Lade divergenz-finetuned PDE-Transformer Modell...")
model = PDETransformer.from_pretrained('thuerey-group/pde-transformer', subfolder='mc-s').to(device)
ckpt_path = Path('checkpoints') / 'pde_transformer_finetuned_divergence_best.pt'
if ckpt_path.exists():
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Divergenz-finetuned Gewichte geladen: {ckpt_path}")
else:
    print("WARNUNG: Keine divergenz-finetuned Gewichte gefunden!")
model.eval()


def compute_nrmse(pred, target):
    """Normalized Root Mean Squared Error."""
    mse = np.mean((pred - target) ** 2)
    rmse = np.sqrt(mse)
    nrmse = rmse / (np.std(target) + 1e-8)
    return nrmse


# Rollout-Bereich: Zeitschritt 10 bis 50
start_step = 10
end_step = 99
print(f"\nRollout von Zeitschritt {start_step} (t={time[start_step]:.2f}s) bis {end_step} (t={time[end_step]:.2f}s)")

# Extrahiere Startfelder bei t=start_step
if vel.shape[1] == 2:
    u_init = vel[start_step, 0]
    w_init = vel[start_step, 1]
else:
    u_init = vel[start_step, :, :, 0]
    w_init = vel[start_step, :, :, 1]

# Rollout
num_rollout_steps = end_step - start_step
predictions_u = [u_init.copy()]
predictions_w = [w_init.copy()]
current_u = u_init.copy()
current_w = w_init.copy()

print(f"Starte Rollout mit {num_rollout_steps} Schritten...")

for step in range(num_rollout_steps):
    input_field = np.stack([current_u, current_w], axis=0)[None, ...]
    input_tensor = torch.from_numpy(input_field).float().to(device)
    class_labels = torch.ones((1,), dtype=torch.long, device=device) * 1000

    with torch.no_grad():
        out = model(hidden_states=input_tensor, class_labels=class_labels)
        pred = out.sample.cpu().numpy()[0]

    current_u = pred[0]
    current_w = pred[1] if pred.shape[0] > 1 else np.zeros_like(current_u)
    predictions_u.append(current_u.copy())
    predictions_w.append(current_w.copy())

# Metriken berechnen
print("\n" + "=" * 60)
print("EVALUATION ERGEBNISSE (Divergenz-Finetuned)")
print("=" * 60)

nrmse_u_list = []
nrmse_w_list = []
div_gt_list = []
div_pred_list = []

for i, t_idx in enumerate(range(start_step, end_step + 1)):
    # Ground Truth
    if vel.shape[1] == 2:
        u_gt, w_gt = vel[t_idx, 0], vel[t_idx, 1]
    else:
        u_gt, w_gt = vel[t_idx, :, :, 0], vel[t_idx, :, :, 1]

    # NRMSE
    nrmse_u = compute_nrmse(predictions_u[i], u_gt)
    nrmse_w = compute_nrmse(predictions_w[i], w_gt)
    nrmse_u_list.append(nrmse_u)
    nrmse_w_list.append(nrmse_w)

    # Divergenz
    div_gt = compute_divergence(u_gt, w_gt, x, z)
    div_pred = compute_divergence(predictions_u[i], predictions_w[i], x, z)
    _, max_div_gt = divergence_stats(div_gt)
    _, max_div_pred = divergence_stats(div_pred)
    div_gt_list.append(max_div_gt)
    div_pred_list.append(max_div_pred)

    print(f"t={t_idx:3d}: NRMSE(u)={nrmse_u:.4f}, NRMSE(w)={nrmse_w:.4f} | "
          f"maxDiv GT={max_div_gt:.2e}, Pred={max_div_pred:.2e}")

print("\n" + "-" * 60)
print(f"Gesamt NRMSE(u):  mean={np.mean(nrmse_u_list):.4f}, final={nrmse_u_list[-1]:.4f}")
print(f"Gesamt NRMSE(w):  mean={np.mean(nrmse_w_list):.4f}, final={nrmse_w_list[-1]:.4f}")
print(f"Divergenz GT:     mean={np.mean(div_gt_list):.2e}, max={np.max(div_gt_list):.2e}")
print(f"Divergenz Pred:   mean={np.mean(div_pred_list):.2e}, max={np.max(div_pred_list):.2e}")

# Visualisierung
Path('results').mkdir(exist_ok=True)

# Plot 1: NRMSE über Zeit
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
time_axis = list(range(start_step, end_step + 1))
axes[0].plot(time_axis, nrmse_u_list, label='NRMSE(u)', marker='o')
axes[0].plot(time_axis, nrmse_w_list, label='NRMSE(w)', marker='s')
axes[0].set_xlabel('Zeitschritt')
axes[0].set_ylabel('NRMSE')
axes[0].set_title(f'NRMSE über Rollout (t={start_step} bis t={end_step}) - Divergenz-Finetuned')
axes[0].legend()
axes[0].grid(True)

axes[1].semilogy(time_axis, div_gt_list, label='GT max|div|', linestyle='--', marker='o')
axes[1].semilogy(time_axis, div_pred_list, label='Pred max|div|', marker='s')
axes[1].set_xlabel('Zeitschritt')
axes[1].set_ylabel('Max |Divergenz|')
axes[1].set_title('Divergenz über Rollout - Divergenz-Finetuned')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(Path('results') / 'test_evaluation_metrics_divergence.png', dpi=150)
plt.show()

# Plot 2: Vergleich zu verschiedenen Zeitpunkten
plot_indices = [0, num_rollout_steps // 2, num_rollout_steps]
fig, axes = plt.subplots(3, len(plot_indices), figsize=(4 * len(plot_indices), 10))

for col, pred_idx in enumerate(plot_indices):
    t_idx = start_step + pred_idx
    if vel.shape[1] == 2:
        u_gt = vel[t_idx, 0]
    else:
        u_gt = vel[t_idx, :, :, 0]

    vmin, vmax = u_gt.min(), u_gt.max()

    axes[0, col].imshow(u_gt.T, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
    axes[0, col].set_title(f'GT t={t_idx}')

    axes[1, col].imshow(predictions_u[pred_idx].T, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
    axes[1, col].set_title(f'Pred t={t_idx}\nNRMSE={nrmse_u_list[pred_idx]:.3f}')

    error = predictions_u[pred_idx] - u_gt
    limit = max(1e-5, np.max(np.abs(error)))
    axes[2, col].imshow(error.T, cmap='seismic', origin='lower', vmin=-limit, vmax=limit)
    axes[2, col].set_title(f'Error t={t_idx}')

axes[0, 0].set_ylabel('Ground Truth')
axes[1, 0].set_ylabel('Prediction')
axes[2, 0].set_ylabel('Error')
plt.tight_layout()
plt.savefig(Path('results') / 'test_rollout_comparison_divergence.png', dpi=150)
plt.show()

# Ergebnisse speichern
with open(Path('results') / 'test_evaluation_divergence.txt', 'w') as f:
    f.write(f"Test-Datensatz: {dataset_dir}\n")
    f.write(f"Modell: Divergenz-Finetuned ({ckpt_path})\n")
    f.write(f"Rollout: Zeitschritt {start_step} bis {end_step}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Gesamt NRMSE(u): mean={np.mean(nrmse_u_list):.6f}, final={nrmse_u_list[-1]:.6f}\n")
    f.write(f"Gesamt NRMSE(w): mean={np.mean(nrmse_w_list):.6f}, final={nrmse_w_list[-1]:.6f}\n\n")
    f.write(f"Divergenz GT:   mean={np.mean(div_gt_list):.6e}, max={np.max(div_gt_list):.6e}\n")
    f.write(f"Divergenz Pred: mean={np.mean(div_pred_list):.6e}, max={np.max(div_pred_list):.6e}\n")

print(f"\nErgebnisse gespeichert in 'results/'")