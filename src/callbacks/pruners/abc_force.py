from abc import ABC

from .abc_pruner import ABCPruner


class ABCForcePruner(ABCPruner, ABC):

    def __init__(self, *args, layer_norm=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_norm = layer_norm

    def compute_importance_scores(self, rescore=True):
        importance_scores = {}
        for module, attr in self.target_moduledict.values():
            for name, param in module.named_parameters():
                if f"{attr}_orig" == name:  # need to get the leaf
                    snip_score = (getattr(
                        module, f"{attr}_orig" if rescore else attr
                    ) * param.grad).abs()
                    break
            else:
                raise ValueError(f"Attribute {attr} not found in module {module}")

            if self.layer_norm:  # Normalize each layer
                snip_score = snip_score / snip_score.norm()

            importance_scores[(module, attr)] = snip_score

        return importance_scores
