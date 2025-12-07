import torch
import numpy as np
from pathlib import Path
from pdetransformer.core.mixed_channels import PDETransformer
from src.data_loader import find_first_h5, load_shear_velocity
from src.divergence_utils import compute_divergence, divergence_stats
import matplotlib.pyplot as plt

# Gerät
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Test-Datensätze aus data_splits.txt laden
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


def compute_nrmse(pred, target):
    """Normalized Root Mean Squared Error."""
    mse = np.mean((pred - target) ** 2)
    rmse = np.sqrt(mse)
    nrmse = rmse / (np.std(target) + 1e-8)
    return nrmse


def run_rollout(model, vel, start_step, end_step):
    """Führt Rollout durch und berechnet Metriken."""
    if vel.shape[1] == 2:
        u_init = vel[start_step, 0]
        w_init = vel[start_step, 1]
    else:
        u_init = vel[start_step, :, :, 0]
        w_init = vel[start_step, :, :, 1]

    num_rollout_steps = end_step - start_step
    predictions_u = [u_init.copy()]
    predictions_w = [w_init.copy()]
    current_u = u_init.copy()
    current_w = w_init.copy()

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
    nrmse_u_list = []
    nrmse_w_list = []
    div_gt_list = []
    div_pred_list = []

    for i, t_idx in enumerate(range(start_step, end_step + 1)):
        if vel.shape[1] == 2:
            u_gt, w_gt = vel[t_idx, 0], vel[t_idx, 1]
        else:
            u_gt, w_gt = vel[t_idx, :, :, 0], vel[t_idx, :, :, 1]

        nrmse_u_list.append(compute_nrmse(predictions_u[i], u_gt))
        nrmse_w_list.append(compute_nrmse(predictions_w[i], w_gt))

        div_gt = compute_divergence(u_gt, w_gt, x, z)
        div_pred = compute_divergence(predictions_u[i], predictions_w[i], x, z)
        _, max_div_gt = divergence_stats(div_gt)
        _, max_div_pred = divergence_stats(div_pred)
        div_gt_list.append(max_div_gt)
        div_pred_list.append(max_div_pred)

    return {
        'nrmse_u': nrmse_u_list,
        'nrmse_w': nrmse_w_list,
        'div_gt': div_gt_list,
        'div_pred': div_pred_list
    }


# Rollout-Bereich
start_step = 10
end_step = 99
print(f"\nRollout von Zeitschritt {start_step} bis {end_step}")

# Modell 1: Default Finetuned (nur MSE)
print("\n--- Lade Default-Finetuned Modell (MSE only) ---")
model_default = PDETransformer.from_pretrained('thuerey-group/pde-transformer', subfolder='mc-s').to(device)
ckpt_default = Path('checkpoints') / 'pde_transformer_finetuned_best.pt'
if ckpt_default.exists():
    state = torch.load(ckpt_default, map_location=device, weights_only=True)
    model_default.load_state_dict(state)
    print(f"Gewichte geladen: {ckpt_default}")
else:
    print("WARNUNG: Keine Default-Gewichte gefunden!")
model_default.eval()

print("Starte Rollout für Default-Modell...")
results_default = run_rollout(model_default, vel, start_step, end_step)

# Modell 2: Divergenz Finetuned (MSE + Divergenz-Loss)
print("\n--- Lade Divergenz-Finetuned Modell (MSE + Div) ---")
model_div = PDETransformer.from_pretrained('thuerey-group/pde-transformer', subfolder='mc-s').to(device)
ckpt_div = Path('checkpoints') / 'pde_transformer_finetuned_divergence_best.pt'
if ckpt_div.exists():
    state = torch.load(ckpt_div, map_location=device, weights_only=True)
    model_div.load_state_dict(state)
    print(f"Gewichte geladen: {ckpt_div}")
else:
    print("WARNUNG: Keine Divergenz-Gewichte gefunden!")
model_div.eval()

print("Starte Rollout für Divergenz-Modell...")
results_div = run_rollout(model_div, vel, start_step, end_step)

# Ergebnisse ausgeben
print("\n" + "=" * 70)
print("VERGLEICH: Default (MSE) vs. Divergenz (MSE + Div-Loss)")
print("=" * 70)

print(f"\n{'Metrik':<25} {'Default':<20} {'Divergenz':<20}")
print("-" * 70)
print(f"{'NRMSE(u) mean':<25} {np.mean(results_default['nrmse_u']):<20.4f} {np.mean(results_div['nrmse_u']):<20.4f}")
print(f"{'NRMSE(u) final':<25} {results_default['nrmse_u'][-1]:<20.4f} {results_div['nrmse_u'][-1]:<20.4f}")
print(f"{'NRMSE(w) mean':<25} {np.mean(results_default['nrmse_w']):<20.4f} {np.mean(results_div['nrmse_w']):<20.4f}")
print(f"{'NRMSE(w) final':<25} {results_default['nrmse_w'][-1]:<20.4f} {results_div['nrmse_w'][-1]:<20.4f}")
print(f"{'Divergenz Pred mean':<25} {np.mean(results_default['div_pred']):<20.2e} {np.mean(results_div['div_pred']):<20.2e}")
print(f"{'Divergenz Pred max':<25} {np.max(results_default['div_pred']):<20.2e} {np.max(results_div['div_pred']):<20.2e}")

# Visualisierung - Separate Plots
Path('results').mkdir(exist_ok=True)
time_axis = list(range(start_step, end_step + 1))

# Plot 1: NRMSE(u)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time_axis, results_default['nrmse_u'], label='Default (MSE)', marker='o', markersize=3)
ax.plot(time_axis, results_div['nrmse_u'], label='Divergenz (MSE+Div)', marker='s', markersize=3)
ax.set_xlabel('Zeitschritt')
ax.set_ylabel('NRMSE(u)')
ax.set_title('NRMSE(u) - Horizontale Geschwindigkeit')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(Path('results') / 'comparison_nrmse_u.png', dpi=150)
plt.show()

# Plot 2: NRMSE(w)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time_axis, results_default['nrmse_w'], label='Default (MSE)', marker='o', markersize=3)
ax.plot(time_axis, results_div['nrmse_w'], label='Divergenz (MSE+Div)', marker='s', markersize=3)
ax.set_xlabel('Zeitschritt')
ax.set_ylabel('NRMSE(w)')
ax.set_title('NRMSE(w) - Vertikale Geschwindigkeit')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(Path('results') / 'comparison_nrmse_w.png', dpi=150)
plt.show()

# Plot 3: Divergenz
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(time_axis, results_default['div_gt'], label='GT', linestyle='--', color='gray')
ax.semilogy(time_axis, results_default['div_pred'], label='Default (MSE)', marker='o', markersize=3)
ax.semilogy(time_axis, results_div['div_pred'], label='Divergenz (MSE+Div)', marker='s', markersize=3)
ax.set_xlabel('Zeitschritt')
ax.set_ylabel('Max |Divergenz|')
ax.set_title('Divergenz über Rollout')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(Path('results') / 'comparison_divergence.png', dpi=150)
plt.show()

# Ergebnisse speichern
with open(Path('results') / 'comparison_default_vs_divergence.txt', 'w') as f:
    f.write(f"Test-Datensatz: {dataset_dir}\n")
    f.write(f"Rollout: Zeitschritt {start_step} bis {end_step}\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"{'Metrik':<25} {'Default':<20} {'Divergenz':<20}\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'NRMSE(u) mean':<25} {np.mean(results_default['nrmse_u']):<20.6f} {np.mean(results_div['nrmse_u']):<20.6f}\n")
    f.write(f"{'NRMSE(u) final':<25} {results_default['nrmse_u'][-1]:<20.6f} {results_div['nrmse_u'][-1]:<20.6f}\n")
    f.write(f"{'NRMSE(w) mean':<25} {np.mean(results_default['nrmse_w']):<20.6f} {np.mean(results_div['nrmse_w']):<20.6f}\n")
    f.write(f"{'NRMSE(w) final':<25} {results_default['nrmse_w'][-1]:<20.6f} {results_div['nrmse_w'][-1]:<20.6f}\n")
    f.write(f"{'Divergenz Pred mean':<25} {np.mean(results_default['div_pred']):<20.6e} {np.mean(results_div['div_pred']):<20.6e}\n")
    f.write(f"{'Divergenz Pred max':<25} {np.max(results_default['div_pred']):<20.6e} {np.max(results_div['div_pred']):<20.6e}\n")

print(f"\nErgebnisse gespeichert in 'results/'")