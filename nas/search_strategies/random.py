from .base import BaseStrategy
import logging
from .build import STRATEGY_REGISTRY


@STRATEGY_REGISTRY.register()
class Random(BaseStrategy):
    def __init__(self, cfg, search_space, device):
        BaseStrategy.__init__(self, cfg, search_space, device)
        self.logger = logging.getLogger(f"nas_gpu_{device}")

    def do_search(self):
        data = self.search_space.random_architectures(num=self.cfg.SEARCH_STRATEGY.SEARCH_BUDGET)
        if self.cfg.SEARCH_SPACE.HAS_GT == "N":
            data = self.search_space.eval_architectures(data)
        top_5_loss = sorted(self.search_space.get_arch_acc_from_arch_list(data))[:min(5, len(data))]
        self.logger.info('Query {}, top 5 val losses {}'.format(self.cfg.SEARCH_STRATEGY.SEARCH_BUDGET,
                                                                top_5_loss))
        return data

    def update_agent(self):
        pass

    def pred_acc_agent(self, *vargs, **kargs):
        pass

    def get_best_candidate_architectures(self, *vargs, **kargs):
        pass