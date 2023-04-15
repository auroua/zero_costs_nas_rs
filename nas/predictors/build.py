import logging
from fvcore.common.registry import Registry

# predictor registry
PREDICTOR_REGISTRY = Registry('PREDICTOR')


def build_predictor(cfg, device, verbose=False):
    logger = logging.getLogger(f"nas_gpu_{device}")
    try:
       predictor = PREDICTOR_REGISTRY.get(cfg.PREDICTOR.TYPE)(cfg)
       if verbose:
           logger.info(f'Build predictor ============= {cfg.PREDICTOR.TYPE} ================')
    except KeyError:
        raise NotImplementedError(f'The predictor type {cfg.PREDICTOR.TYPE} have not implemented!')
    return predictor
