from __future__ import annotations

import keras
import numpy as np


class StrategyManager:
    """Applies a freezing policy to a backbone *before* the model is compiled.

    Strategies
    ----------
    feature_extraction : whole backbone frozen (the builder also runs it in inference mode).
    partial_ft         : last `unfreeze_fraction` of the backbone's parameters are trainable.
    full_ft            : whole backbone trainable.

    BatchNorm: a BatchNormalization layer with ``trainable=False`` runs in inference
    mode (moving statistics are not updated), so ``freeze_bn=True`` keeps the
    pretrained statistics intact. This is the usual recommendation for small batches.
    """

    STRATEGIES = ("feature_extraction", "partial_ft", "full_ft")

    @staticmethod
    def _count_params(weights: list) -> int:
        return sum(int(np.prod(w.shape)) for w in weights)

    @classmethod
    def _leaf_layers(cls, model: keras.Model) -> list[keras.layers.Layer]:
        """Flattened list of non-container layers (handles nested sub-models)."""
        leaves: list[keras.layers.Layer] = []
        for layer in model.layers:
            if hasattr(layer, "layers") and layer.layers:
                leaves.extend(cls._leaf_layers(layer))
            else:
                leaves.append(layer)
        return leaves

    @classmethod
    def apply_strategy(
        cls,
        backbone: keras.Model,
        strategy: str,
        unfreeze_fraction: float = 0.0,
        freeze_bn: bool = True,
    ) -> None:
        if strategy not in cls.STRATEGIES:
            raise ValueError(f"Unknown strategy: '{strategy}'. Expected one of {cls.STRATEGIES}.")

        if strategy == "feature_extraction":
            backbone.trainable = False
            return

        backbone.trainable = True
        leaves = cls._leaf_layers(backbone)
        bn_cls = keras.layers.BatchNormalization

        if strategy == "full_ft":
            if freeze_bn:
                for layer in leaves:
                    if isinstance(layer, bn_cls):
                        layer.trainable = False
            return

        # partial_ft
        if not 0.0 < unfreeze_fraction <= 1.0:
            raise ValueError(f"unfreeze_fraction must be in (0, 1], got {unfreeze_fraction}.")

        for layer in leaves:
            layer.trainable = False

        # Only count layers that *can* be unfrozen, so the fraction is meaningful.
        eligible = [
            l for l in leaves
            if l.weights and not (freeze_bn and isinstance(l, bn_cls))
        ]
        target = sum(cls._count_params(l.weights) for l in eligible) * unfreeze_fraction

        unfrozen = 0
        for layer in reversed(eligible):
            layer.trainable = True
            unfrozen += cls._count_params(layer.weights)
            if unfrozen >= target:
                break

    @classmethod
    def summarize_trainable(cls, model: keras.Model) -> dict:
        trainable = cls._count_params(model.trainable_weights)
        total = cls._count_params(model.weights)
        return {
            "trainable_params": trainable,
            "total_params": total,
            "trainable_fraction": trainable / max(total, 1),
        }