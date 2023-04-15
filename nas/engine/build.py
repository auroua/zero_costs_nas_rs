import logging
from fvcore.common.registry import Registry

# trainer registry
TRAINER_REGISTRY = Registry('TRAINER')


def build_predictor_trainer(cfg, device, num_architectures, verbose=False):
    try:
        logger = logging.getLogger(f"nas_gpu_{device}")
        predictor = TRAINER_REGISTRY.get(cfg.SEARCH_STRATEGY.NPENAS.ENGINE)(
            cfg,
            device=device,
            num_architectures=num_architectures
        )
        if verbose:
            logger.info(f'Build predictor trainer ============= {cfg.SEARCH_STRATEGY.NPENAS.EIGEN} ================')
    except KeyError:
        raise NotImplementedError(f'The trainer {cfg.SEARCH_STRATEGY.NPENAS.EIGEN} have not implemented!')
    return predictor
