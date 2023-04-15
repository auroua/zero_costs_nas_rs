from detectron2.utils.registry import Registry


NAS_LAYER_REGISTRY = Registry("SEG_NAS_LAYERS")
NAS_LAYER_REGISTRY.__doc__ = """ The layers used to construct the segmentation head. """


def build_seg_nas_layers(layer_type, **kvargs):
    return NAS_LAYER_REGISTRY.get(layer_type)(**kvargs)