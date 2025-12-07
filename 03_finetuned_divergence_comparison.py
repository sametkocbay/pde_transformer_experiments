import torch
import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from src.data_loader import find_first_h5, load_shear_velocity
from src.divergence_utils import compute_divergence, divergence_stats
from pdetransformer.core.mixed_channels import PDETransformer

# Gerät
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Datensatz
dataset_dir = str(Path("dataset") / "sf_128x256_reynolds_1_00e+03_schmidt_1_00e+00_width_2_50e-01_nshear_2_nblobs_2")
filename = str(find_first_h5(dataset_dir))

# Lade Daten
d = load_shear_velocity(filename)
vel = d['vel']
time = d['time']
x = d['x']
z = d['z']

# Modell laden und Fine-Tune-Gewichte anwenden
model = PDETransformer.from_pretrained('thuerey-group/pde-transformer', subfolder='mc-s').to(device)
ckpt_path = Path('checkpoints') / 'pde_transformer_finetuned_divergence.pt'
if ckpt_path.exists():
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    print(f"Fine-tuned Gewichte geladen: {ckpt_path}")
else:
    print("Warnung: Keine fine-tuned Gewichte gefunden. Nutze Pretrained.")
model.eval()

# Vergleich über mehrere Zeitpunkte
indices = [1, 10, 20, -1]
for t_idx in indices:
    if t_idx == -1:
        t_idx = len(time) - 1

    # u,w Komponenten
    if vel.shape[1] == 2:
        u = vel[t_idx, 0]
        w = vel[t_idx, 1]
    elif vel.shape[3] == 2:
        u = vel[t_idx, :, :, 0]
        w = vel[t_idx, :, :, 1]
    else:
        raise RuntimeError(f"Unbekannte Velocity-Shape: {vel.shape}")

    div_gt = compute_divergence(u, w, x, z)
    m_gt, M_gt = divergence_stats(div_gt)

    input_field = np.stack([u, w], axis=0)[None, ...]
    input_tensor = torch.from_numpy(input_field).float().to(device)
    class_labels = torch.ones((1,), dtype=torch.long).to(device) * 1000

    with torch.no_grad():
        out = model(hidden_states=input_tensor, class_labels=class_labels)
        pred = out.sample.detach().cpu().numpy()[0]

    u_pred = pred[0]
    w_pred = pred[1] if pred.shape[0] > 1 else np.zeros_like(u_pred)

    div_pred = compute_divergence(u_pred, w_pred, x, z)
    m_p, M_p = divergence_stats(div_pred)

    print(f"t={time[t_idx]:.2f} | GT(mean={m_gt:.2e}, max={M_gt:.2e}) | Pred(mean={m_p:.2e}, max={M_p:.2e})")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im0 = axes[0].imshow(u.T, cmap='RdBu_r', origin='lower'); axes[0].set_title('U Ground Truth'); plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(u_pred.T, cmap='RdBu_r', origin='lower'); axes[1].set_title('U Prediction (finetuned)'); plt.colorbar(im1, ax=axes[1])
    limit = max(1e-5, np.max(np.abs(div_pred)))
    im2 = axes[2].imshow(div_pred.T, cmap='seismic', origin='lower', vmin=-limit, vmax=limit); axes[2].set_title('Divergenz Prediction'); plt.colorbar(im2, ax=axes[2], format='%.1e')
    plt.tight_layout(); plt.show()

# Ergebnisse speichern
Path('results').mkdir(exist_ok=True)
with open(Path('results')/ 'finetuned_divergence.txt', 'w') as f:
    f.write('Vergleichswerte wurden im Konsolen-Output angezeigt.\n')

