import torch
import numpy as np
from pathlib import Path
from pdetransformer.core.mixed_channels import PDETransformer
from src.losses import CombinedMSEAndDivLoss
from src.data_loader import find_first_h5, load_shear_velocity
from torch.utils.data import Dataset, DataLoader
import random

# Gerät
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Alle Datensatz-Ordner sammeln
dataset_root = Path("dataset")
all_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
print(f"Gefunden: {len(all_dirs)} Datensätze")

# Shuffle und Split (32 Train, 4 Val, 4 Test) - gleicher Seed wie 00_finetune_default.py
random.seed(42)
shuffled_dirs = all_dirs.copy()
random.shuffle(shuffled_dirs)

train_dirs = shuffled_dirs[:32]
val_dirs = shuffled_dirs[32:36]
test_dirs = shuffled_dirs[36:40]

print(f"Train: {len(train_dirs)}, Val: {len(val_dirs)}, Test: {len(test_dirs)}")

# Splits speichern
Path('checkpoints').mkdir(exist_ok=True)
with open(Path('checkpoints') / 'data_splits_divergence.txt', 'w') as f:
    f.write("TRAIN:\n")
    for d in train_dirs:
        f.write(f"  {d.name}\n")
    f.write("\nVAL:\n")
    for d in val_dirs:
        f.write(f"  {d.name}\n")
    f.write("\nTEST:\n")
    for d in test_dirs:
        f.write(f"  {d.name}\n")


class ShearFlowDataset(Dataset):
    """Dataset für Shear-Flow Trajektorien."""

    def __init__(self, data_dirs):
        self.samples = []
        for data_dir in data_dirs:
            try:
                filename = str(find_first_h5(str(data_dir)))
                d = load_shear_velocity(filename)
                vel = d['vel']

                for t in range(len(vel) - 1):
                    if vel.shape[1] == 2:
                        u_in, w_in = vel[t, 0], vel[t, 1]
                        u_out, w_out = vel[t + 1, 0], vel[t + 1, 1]
                    else:
                        u_in, w_in = vel[t, :, :, 0], vel[t, :, :, 1]
                        u_out, w_out = vel[t + 1, :, :, 0], vel[t + 1, :, :, 1]

                    input_field = np.stack([u_in, w_in], axis=0)
                    target_field = np.stack([u_out, w_out], axis=0)
                    self.samples.append((input_field, target_field))
            except Exception as e:
                print(f"Warnung: Konnte {data_dir} nicht laden: {e}")

        print(f"  Geladen: {len(self.samples)} Samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        return torch.from_numpy(inp).float(), torch.from_numpy(tgt).float()


# Datasets erstellen
print("Lade Trainingsdaten...")
train_dataset = ShearFlowDataset(train_dirs)
print("Lade Validierungsdaten...")
val_dataset = ShearFlowDataset(val_dirs)

# DataLoader
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Modell laden
print("Lade vortrainiertes PDE-Transformer Modell...")
model = PDETransformer.from_pretrained('thuerey-group/pde-transformer', subfolder='mc-s').to(device)
model.train()

# Optimierer und Loss mit Divergenz-Regularisierung
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
criterion = CombinedMSEAndDivLoss(lambda_div=0.1, reduction='mean')

# Training
num_epochs = 50
best_val_loss = float('inf')

print(f"\nStarte Training für {num_epochs} Epochen...")
print(f"Batches pro Epoche: {len(train_loader)}")
print(f"Loss: MSE + 0.1 * Divergenz-Loss")

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        class_labels = torch.ones((inputs.size(0),), dtype=torch.long, device=device) * 1000

        optimizer.zero_grad()
        out = model(hidden_states=inputs, class_labels=class_labels)
        pred = out.sample

        loss = criterion(pred, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validierung
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            class_labels = torch.ones((inputs.size(0),), dtype=torch.long, device=device) * 1000

            out = model(hidden_states=inputs, class_labels=class_labels)
            pred = out.sample
            loss = criterion(pred, targets)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    scheduler.step()

    print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

    # Bestes Modell speichern
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), Path('checkpoints') / 'pde_transformer_finetuned_divergence_best.pt')
        print(f"  -> Neues bestes Modell gespeichert (Val Loss: {val_loss:.6f})")

# Finales Modell speichern
torch.save(model.state_dict(), Path('checkpoints') / 'pde_transformer_finetuned_divergence_final.pt')
print(f"\nTraining abgeschlossen. Bester Val Loss: {best_val_loss:.6f}")