from fvcore.common.registry import Registry


# search space registry
SPACE_REGISTRY = Registry('SEARCH_SPACE')


def build_search_space(cfg):
    try:
        ss = SPACE_REGISTRY.get(cfg.SEARCH_SPACE.TYPE)(cfg)
    except KeyError:
        raise NotImplementedError(f'The search space {cfg.SEARCH_SPACE.TYPE} have not implemented!')
    return ss
