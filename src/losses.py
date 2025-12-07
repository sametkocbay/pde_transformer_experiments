import torch
import torch.nn as nn

class DivergenceLoss(nn.Module):
    """
    Verlust, der die Divergenz (du/dx + dw/dz) eines 2D-Vektorfelds (u, w) bestraft.
    Erwartet Tensoren mit Shape (B, 2, X, Z) für pred und target.
    Optional kann auch nur pred übergeben werden, dann wird absolute Divergenz minimiert.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        # pred: (B, 2, X, Z)
        u = pred[:, 0]
        w = pred[:, 1]
        # zentrale Differenzen via torch.gradient
        du_dx = torch.gradient(u, dim=1)[0]
        dw_dz = torch.gradient(w, dim=2)[0]
        div = du_dx + dw_dz
        if target is not None:
            u_t = target[:, 0]
            w_t = target[:, 1]
            du_dx_t = torch.gradient(u_t, dim=1)[0]
            dw_dz_t = torch.gradient(w_t, dim=2)[0]
            div_t = du_dx_t + dw_dz_t
            loss = (div - div_t) ** 2
        else:
            loss = div ** 2
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class CombinedMSEAndDivLoss(nn.Module):
    """
    Kombination aus MSE auf den Feldern und Divergenz-Verlust.
    """
    def __init__(self, lambda_div: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.lambda_div = lambda_div
        self.mse = nn.MSELoss(reduction=reduction)
        self.div_loss = DivergenceLoss(reduction=reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(pred, target) + self.lambda_div * self.div_loss(pred, target)

