from abc import ABC

from .abc_pruner import ABCPruner


class ABCMagnitudePruner(ABCPruner, ABC):

    def __init__(self, *args, layer_norm=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_norm = layer_norm

    def compute_importance_scores(self, rescore=False):
        importance_scores = {}
        for module, attr in self.target_moduledict.values():
            if rescore:
                for name, param in module.named_parameters():
                    if f"{attr}_orig" == name:  # need to get the leaf
                        importance_score = getattr(module, f"{attr}_orig").abs()
                        break
                else:
                    raise ValueError(f"Attribute {attr} not found in module {module}")
            else:
                weights = getattr(module, attr)
                importance_score = weights.abs()

            if self.layer_norm:  # Normalize each layer
                importance_score = importance_score / importance_score.norm()

            importance_scores[(module, attr)] = importance_score

        return importance_scores
