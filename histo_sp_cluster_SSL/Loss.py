import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class BaseContrastiveLoss(nn.Module):
    """
    Base class for contrastive loss functions.
    """

    def __init__(self):
        super(BaseContrastiveLoss, self).__init__()

    def loss_original(self, q: torch.Tensor, k: torch.Tensor, queue: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Compute the original contrastive loss (MoCo-style).

        Args:
            q: (N, D) query embeddings
            k: (N, D) positive key embeddings
            queue: (K, D) memory bank of negative embeddings
            temperature: temperature for scaling logits

        Returns:
            Contrastive loss as a scalar tensor.
        """
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)  # (N, 1)
        l_neg = torch.einsum("nc,ck->nk", [q, queue.clone().detach().T])  # (N, K)

        logits = torch.cat([l_pos, l_neg], dim=1)  # (N, 1+K)
        logits /= temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss

class ContrastiveLoss(BaseContrastiveLoss):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, q: torch.Tensor, k: torch.Tensor, queue: torch.Tensor) -> torch.Tensor:
        return self.loss_original(q, k, queue, self.temperature)


class ClusterLoss(nn.Module):
    """
    Contrastive loss + false-negative + neighbor-based regularization.
    """

    def __init__(self, temperature: float = 0.07, alpha: float = 0.4, beta: float = 0.2, lambda_bml: float = 0.2):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.lambda_bml = lambda_bml

    def forward(self, q, k, k2, hard_negatives, false_negatives):
        """
        Compute the full ClusterLoss.

        Args:
            q: (B, D) query embeddings
            k: (B, D) positive key embeddings
            k2: (B, D) neighbor key embeddings
            hard_negatives: list of (N_i, D) tensors
            false_negatives: list of (M_i, D) tensors

        Returns:
            Tuple of (total loss, contrastive loss, fn_bml loss, nb_bml loss)
        """
        contrastive_losses = []
        contrastive_losses_neighbor = []
        bml_fn, bml_nb = [], []

        for i in range(q.shape[0]):
            qi, ki, k2i = q[i].unsqueeze(0), k[i].unsqueeze(0), k2[i].unsqueeze(0)
            hn_i, fn_i = hard_negatives[i], false_negatives[i]

            if hn_i.shape[0] == 0:
                continue

            # Contrastive loss
            l_pos = torch.einsum("nc,nc->n", qi, ki).unsqueeze(1)
            l_neg = torch.matmul(qi, hn_i.T)
            logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
            label = torch.zeros(1, dtype=torch.long, device=q.device)
            contrastive_losses.append(F.cross_entropy(logits, label))

            # Contrastive neighbor loss
            l_pos_neighbor = torch.einsum("nc,nc->n", qi, k2i).unsqueeze(1)
            #l_neg_neighbor = torch.matmul(qi, hn_i.T)
            logits_neighbor = torch.cat([l_pos_neighbor, l_neg], dim=1) / self.temperature
            label_neighbor = torch.zeros(1, dtype=torch.long, device=q.device)
            contrastive_losses_neighbor.append(F.cross_entropy(logits_neighbor, label_neighbor))

            # BML (False Negative)
            if fn_i.shape[0] > 0:
                sim_fn = torch.einsum("nd,md->", qi, fn_i) / fn_i.shape[0]
                sim_pos = torch.einsum("nd,nd->", qi, ki)
                delta_fn = sim_fn - sim_pos
                bml_fn.append(F.relu(delta_fn + self.alpha) + F.relu(-delta_fn - self.beta))

            """
            # BML (Neighbor)
            sim_nb = torch.einsum("nd,nd->", qi, k2i)
            sim_pos = torch.einsum("nd,nd->", qi, ki)
            delta_nb = sim_nb - sim_pos
            bml_nb.append(F.relu(delta_nb + self.alpha) + F.relu(-delta_nb - self.beta))
            """


        if not contrastive_losses:
            zero = (q.sum() * 0.0).clone()
            return zero, zero, zero, zero

        contrastive_loss = torch.stack(contrastive_losses).mean()
        contrastive_loss_neighbor = torch.stack(contrastive_losses_neighbor).mean()
        bml_fn_term = torch.stack(bml_fn).mean() if bml_fn else torch.tensor(0.0, device=q.device)
        #bml_nb_term = torch.stack(bml_nb).mean() if bml_nb else torch.tensor(0.0, device=q.device)
        #total_bml = 0.5 * self.lambda_bml * (bml_fn_term + bml_nb_term)
        #total_bml = 0.5 * self.lambda_bml * (bml_fn_term + contrastive_loss_neighbor)
        tot_loss = contrastive_loss + contrastive_loss_neighbor + self.lambda_bml * bml_fn_term

        return tot_loss, contrastive_loss, self.lambda_bml * bml_fn_term, contrastive_loss_neighbor