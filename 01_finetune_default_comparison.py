import torch
import numpy as np
import h5py
from pathlib import Path
from pdetransformer.core.mixed_channels import PDETransformer
from src.data_loader import find_first_h5, load_shear_velocity
from src.divergence_utils import compute_divergence, divergence_stats
import matplotlib.pyplot as plt

# Gerät
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Datensatz
dataset_dir = str(Path("dataset") / "sf_128x256_reynolds_1_00e+03_schmidt_1_00e+00_width_2_50e-01_nshear_2_nblobs_2")
filename = str(find_first_h5(dataset_dir))

# Lade Velocity-Feld und Koordinaten
d = load_shear_velocity(filename)
vel = d['vel']
time = d['time']
x = d['x']
z = d['z']

# Modell
print("Lade vortrainiertes PDE-Transformer Modell...")
model = PDETransformer.from_pretrained('thuerey-group/pde-transformer', subfolder='mc-s').to(device)
model.eval()

# Rollout-Parameter
num_rollout_steps = len(time) - 1  # Anzahl der Vorhersage-Schritte

# Extrahiere Startfelder bei t=0
if vel.ndim == 4:
    if vel.shape[1] == 2:
        u_init = vel[0, 0]
        w_init = vel[0, 1]
    elif vel.shape[3] == 2:
        u_init = vel[0, :, :, 0]
        w_init = vel[0, :, :, 1]
    else:
        raise RuntimeError(f"Unbekannte Velocity-Shape: {vel.shape}")
else:
    raise RuntimeError(f"Unerwartete Dimensionen: {vel.shape}")

# Speicher für Rollout-Ergebnisse
predictions_u = [u_init.copy()]
predictions_w = [w_init.copy()]
divergences = []

# Aktueller Zustand für Rollout
current_u = u_init.copy()
current_w = w_init.copy()

print(f"Starte Rollout mit {num_rollout_steps} Schritten...")

# Rollout-Schleife
for step in range(num_rollout_steps):
    # Input vorbereiten
    input_field = np.stack([current_u, current_w], axis=0)[None, ...]  # (B=1, C=2, X, Z)
    input_tensor = torch.from_numpy(input_field).float().to(device)
    class_labels = torch.ones((1,), dtype=torch.long).to(device) * 1000  # unbekannte Klasse

    # Vorhersage
    with torch.no_grad():
        out = model(hidden_states=input_tensor, class_labels=class_labels)
        pred = out.sample.detach().cpu().numpy()[0]  # (C, X, Z)

    # Extrahiere u, w
    current_u = pred[0]
    current_w = pred[1] if pred.shape[0] > 1 else np.zeros_like(current_u)

    predictions_u.append(current_u.copy())
    predictions_w.append(current_w.copy())

    # Divergenz berechnen
    div_pred = compute_divergence(current_u, current_w, x, z)
    mean_div, max_div = divergence_stats(div_pred)
    divergences.append((mean_div, max_div))

    if step % 10 == 0 or step == num_rollout_steps - 1:
        print(f"  Step {step + 1}/{num_rollout_steps}: div mean={mean_div:.2e}, maxAbs={max_div:.2e}")

# Ground-Truth Divergenz für Vergleich
gt_divergences = []
for t_idx in range(len(time)):
    if vel.shape[1] == 2:
        u_gt = vel[t_idx, 0]
        w_gt = vel[t_idx, 1]
    else:
        u_gt = vel[t_idx, :, :, 0]
        w_gt = vel[t_idx, :, :, 1]
    div_gt = compute_divergence(u_gt, w_gt, x, z)
    gt_divergences.append(divergence_stats(div_gt))

# Visualisierung: Vergleich zu verschiedenen Zeitpunkten
plot_steps = [0, len(time) // 4, len(time) // 2, 3 * len(time) // 4, len(time) - 1]
fig, axes = plt.subplots(3, len(plot_steps), figsize=(4 * len(plot_steps), 10))

for col, t_idx in enumerate(plot_steps):
    # GT
    if vel.shape[1] == 2:
        u_gt = vel[t_idx, 0]
    else:
        u_gt = vel[t_idx, :, :, 0]

    vmin, vmax = u_gt.min(), u_gt.max()

    axes[0, col].imshow(u_gt.T, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
    axes[0, col].set_title(f'GT t={t_idx}')

    axes[1, col].imshow(predictions_u[t_idx].T, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
    axes[1, col].set_title(f'Pred t={t_idx}')

    # Fehler
    error = predictions_u[t_idx] - u_gt
    limit = max(1e-5, np.max(np.abs(error)))
    axes[2, col].imshow(error.T, cmap='seismic', origin='lower', vmin=-limit, vmax=limit)
    axes[2, col].set_title(f'Error t={t_idx}')

axes[0, 0].set_ylabel('Ground Truth')
axes[1, 0].set_ylabel('Prediction')
axes[2, 0].set_ylabel('Error')
plt.tight_layout()
plt.savefig(Path('results') / 'rollout_comparison.png', dpi=150)
plt.show()

# Divergenz über Zeit plotten
fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy([d[1] for d in gt_divergences], label='GT maxAbs(div)', linestyle='--')
ax.semilogy([d[1] for d in divergences], label='Pred maxAbs(div)')
ax.set_xlabel('Zeitschritt')
ax.set_ylabel('Max |Divergenz|')
ax.legend()
ax.set_title('Divergenz-Entwicklung über Rollout')
plt.tight_layout()
plt.savefig(Path('results') / 'divergence_over_time.png', dpi=150)
plt.show()

# Speichere Metriken
Path('results').mkdir(exist_ok=True)
with open(Path('results') / 'baseline_divergence.txt', 'w') as f:
    f.write("Rollout Divergenz-Metriken\n")
    f.write("=" * 50 + "\n")
    for step, (mean_div, max_div) in enumerate(divergences):
        f.write(f"Step {step + 1}: mean={mean_div:.6e}, maxAbs={max_div:.6e}\n")