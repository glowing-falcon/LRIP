import torch
import torch.nn as nn


class TopKAccuracy(nn.Module):
    def __init__(self, k: int):
        super(TopKAccuracy, self).__init__()
        self.k = k

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the Top-K accuracy.

        Args:
            outputs (torch.Tensor): Model output logits of shape (batch_size, num_classes).
            targets (torch.Tensor): True labels of shape (batch_size, num_classes if cutmix is used).

        Returns:
            torch.Tensor: Top-K accuracy as a fraction.
        """
        with torch.no_grad():
            _, topk_indices = outputs.topk(self.k, dim=1, largest=True, sorted=True)
            if targets.shape == outputs.shape:
                # If targets are one-hot encoded (e.g., due to cutmix), convert to class indices
                targets = targets.argmax(dim=1)
            topk_correct = topk_indices.eq(targets.view(-1, 1).expand_as(topk_indices))
            topk_accuracy = topk_correct.any(dim=1).float().mean()
        return topk_accuracy
