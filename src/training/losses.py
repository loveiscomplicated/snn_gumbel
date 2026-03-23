"""
Loss functions delegated from SNNModel.
These are thin wrappers so callers can reference losses without importing the model.
"""

import torch
import torch.nn as nn

ce_loss = nn.CrossEntropyLoss()


def total_loss(rates, labels, model, lambda_sparse: float, lambda_commit: float) -> torch.Tensor:
    loss = ce_loss(rates, labels)
    sp_loss = model.sparsity_loss()
    cm_loss = model.commitment_loss()
    # sparsity/commitment terms are zero tensors when no learned layers exist
    if isinstance(sp_loss, float):
        sp_loss = torch.tensor(sp_loss, device=rates.device)
    if isinstance(cm_loss, float):
        cm_loss = torch.tensor(cm_loss, device=rates.device)
    return loss + lambda_sparse * sp_loss + lambda_commit * cm_loss
