import keras
import numpy as np

class StrategyManager:
    @staticmethod
    def _count_params(weights: list) -> int:
        return sum(int(np.prod(w.shape)) for w in weights)

    @classmethod
    def apply_strategy(cls, backbone: keras.Model, strategy: str, unfreeze_fraction: float = 0.0) -> None:
        if strategy == "feature_extraction":
            backbone.trainable = False
        elif strategy == "full_ft":
            backbone.trainable = True
        elif strategy == "partial_ft":
            backbone.trainable = True
            for layer in backbone.layers:
                layer.trainable = False

            target_params = cls._count_params(backbone.weights) * unfreeze_fraction
            unfrozen_params = 0

            for layer in reversed(backbone.layers):
                if not layer.weights: continue
                if not isinstance(layer, keras.layers.BatchNormalization):
                    layer.trainable = True
                
                unfrozen_params += cls._count_params(layer.weights)
                if unfrozen_params >= target_params: break
        else:
            raise ValueError(f"Unknown strategy: '{strategy}'")

    @classmethod
    def summarize_trainable(cls, model: keras.Model) -> dict:
        trainable = cls._count_params(model.trainable_weights)
        total = cls._count_params(model.weights)
        return {
            "trainable_params": trainable,
            "total_params": total,
            "trainable_fraction": trainable / max(total, 1),
        }