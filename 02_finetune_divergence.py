import torch
import numpy as np
from pathlib import Path
from pdetransformer.core.mixed_channels import PDETransformer
from src.losses import CombinedMSEAndDivLoss
from src.data_loader import find_first_h5, load_shear_velocity

# Gerät
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Modell laden
model = PDETransformer.from_pretrained('thuerey-group/pde-transformer', subfolder='mc-s').to(device)
model.train()

# Optimierer und Verlust
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = CombinedMSEAndDivLoss(lambda_div=1.0, reduction='mean')

# Daten laden (einfaches, kleines Feintuning-Demo)
dataset_dir = str(Path("dataset") / "sf_128x256_reynolds_1_00e+03_schmidt_1_00e+00_width_2_50e-01_nshear_2_nblobs_2")
filename = str(find_first_h5(dataset_dir))
d = load_shear_velocity(filename)
vel = d['vel']  # (T, C?, X, Z)
time = d['time']

# Wir verwenden einige Zeitpunkte als Trainingsdaten
train_indices = list(range(0, min(20, len(time))))

for epoch in range(3):  # kurz halten
    total_loss = 0.0
    for t_idx in train_indices:
        # Komponenten u,w extrahieren
        if vel.shape[1] == 2:
            u = vel[t_idx, 0]
            w = vel[t_idx, 1]
        elif vel.shape[3] == 2:
            u = vel[t_idx, :, :, 0]
            w = vel[t_idx, :, :, 1]
        else:
            raise RuntimeError(f"Unbekannte Velocity-Shape: {vel.shape}")

        input_field = np.stack([u, w], axis=0)[None, ...]  # (B=1, C=2, X, Z)
        input_tensor = torch.from_numpy(input_field).float().to(device)
        target_tensor = input_tensor.clone()  # nächster Schritt unbekannt -> hier Identität zum Demonstrieren

        class_labels = torch.ones((1,), dtype=torch.long).to(device) * 1000

        optimizer.zero_grad()
        out = model(hidden_states=input_tensor, class_labels=class_labels)
        pred = out.sample  # (B, C, X, Z)

        loss = criterion(pred, target_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}: loss={total_loss/len(train_indices):.6f}")

# Gewichte speichern
Path('checkpoints').mkdir(exist_ok=True)
ckpt_path = Path('checkpoints') / 'pde_transformer_finetuned_divergence.pt'
torch.save(model.state_dict(), ckpt_path)
print(f"Gespeichert: {ckpt_path}")

