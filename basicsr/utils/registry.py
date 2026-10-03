"""Minimal registry shim."""

class Registry:
    def __init__(self, name):
        self.name = name
        self._obj_map = {}

    def register(self, obj=None, name=None):
        if obj is None:
            def wrapper(fn_or_class):
                n = name or fn_or_class.__name__
                self._obj_map[n] = fn_or_class
                return fn_or_class
            return wrapper
        n = name or obj.__name__
        self._obj_map[n] = obj
        return obj

    def get(self, name):
        return self._obj_map.get(name)

    def __contains__(self, name):
        return name in self._obj_map

ARCH_REGISTRY = Registry('arch')
MODEL_REGISTRY = Registry('model')
LOSS_REGISTRY = Registry('loss')
METRIC_REGISTRY = Registry('metric')
DATASET_REGISTRY = Registry('dataset')
