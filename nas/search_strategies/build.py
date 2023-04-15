from fvcore.common.registry import Registry

# search strategy registry
STRATEGY_REGISTRY = Registry('SEARCH_STRATEGY')


def build_search_strategy(cfg, search_space, device=None):
    try:
        strategy = STRATEGY_REGISTRY.get(cfg.SEARCH_STRATEGY.TYPE)(cfg,
                                                                   search_space=search_space,
                                                                   device=device)
    except KeyError:
        raise NotImplementedError(f'The search strategy {cfg.SEARCH_STRATEGY.TYPE} have not implemented!')
    return strategy
