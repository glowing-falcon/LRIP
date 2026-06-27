from abc import ABC

from .abc_pruner import ABCPruner


class ABCForceMagnitudePruner(ABCPruner, ABC):

    def __init__(self, *args, layer_norm=False, prune_alpha=0.001, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_norm = layer_norm
        self.prune_alpha = prune_alpha

    def compute_importance_scores(self, rescore=True, beta=1.0):
        importance_scores = {}
        for module, attr in self.target_moduledict.values():
            for name, param in module.named_parameters():
                if f"{attr}_orig" == name:  # need to get the leaf
                    mag_score = getattr(module, f"{attr}_orig" if rescore else attr).abs()
                    snip_score = (mag_score * param.grad).abs()
                    break
            else:
                raise ValueError(f"Attribute {attr} not found in module {module}")

            importance_score = (beta * snip_score) + (self.prune_alpha * (mag_score ** 2))
            if self.layer_norm:  # Normalize each layer
                importance_score = importance_score / importance_score.norm()

            importance_scores[(module, attr)] = importance_score

        return importance_scores
