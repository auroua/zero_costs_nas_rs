from .base import BaseStrategy
from nas.engine.build import build_predictor_trainer
from nas.search_strategies.build import STRATEGY_REGISTRY
import logging
import numpy as np
import copy


@STRATEGY_REGISTRY.register()
class NPENAS(BaseStrategy):
    def __init__(self, cfg, search_space, device=None) -> None:
        BaseStrategy.__init__(self, cfg, search_space, device)
        if device:
            self.logger = logging.getLogger(f"nas_gpu_{device}")
        else:
            self.logger = logging.getLogger(__name__)

    def do_search(self):
        data = self.search_space.random_architectures(num=self.cfg.SEARCH_STRATEGY.NUM_INIT)
        data = self.search_space.eval_architectures(data)
        query = self.cfg.SEARCH_STRATEGY.NUM_INIT + self.cfg.SEARCH_STRATEGY.K
        train_data = []
        self.logger.info(f'============= Neural Predictor Trainer: {self.cfg.SEARCH_STRATEGY.NPENAS.ENGINE}, '
                         f'Neural Predictor: {self.cfg.PREDICTOR.TYPE} ================')
        while query <= self.cfg.SEARCH_STRATEGY.SEARCH_BUDGET:
            if len(train_data) <= self.cfg.SEARCH_STRATEGY.FIXED_NUM:
                train_data = copy.deepcopy(data)
                train_flag = True
            else:
                train_flag = False

            if train_flag:
                # build predictor
                predictor_trainer = build_predictor_trainer(
                    self.cfg,
                    device=self.device,
                    num_architectures=len(train_data)
                )
                val_error = self.search_space.get_arch_acc_from_arch_list(train_data)
                # process neural architecture for training
                all_g_data = [arch.g_data() for arch in train_data]
                # update predictor
                self.update_agent(predictor_trainer, all_g_data, val_error)
                train_arch_acc = self.pred_acc_agent(predictor_trainer, all_g_data)
            # generate candidate neural architectures
            candidates = self.search_space.get_candidates(
                data=data,
                num=self.cfg.SEARCH_STRATEGY.CANDIDATE_NUMS,
                num_best_arches=self.cfg.SEARCH_STRATEGY.NUM_BEST_ARCHITECTURES
            )
            # pred the performance of candidate architectures
            candidate_g_data = [d.g_data() for d in candidates]
            candidate_acc = self.pred_acc_agent(predictor_trainer, candidate_g_data)
            # get the best performing neural architectures based on the error predicted by the agent
            if self.cfg.SEARCH_SPACE.TYPE == "SEG101":
                indices = self.get_best_candidate_architectures(candidate_acc, reverse=True)
            else:
                indices = self.get_best_candidate_architectures(candidate_acc)
            arch_best = [candidates[i] for i in indices]
            arch_best = self.search_space.eval_architectures(arch_best)
            data.extend(arch_best)

            if self.cfg.SEARCH_SPACE.TYPE == "SEG101":
                top_5_loss = sorted(self.search_space.get_arch_acc_from_arch_list(data), reverse=True)[
                             :min(5, len(data))]
            else:
                top_5_loss = sorted(self.search_space.get_arch_acc_from_arch_list(data))[
                             :min(5, len(data))]
            if len(train_arch_acc) > 1:
                mean_error = np.mean(np.abs(train_arch_acc-np.array(val_error)))
                self.logger.info('Query {}, training mean loss is {}'.format(
                    query, mean_error))
            self.logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
            query += self.cfg.SEARCH_STRATEGY.K
        return data

    def update_agent(self, predictor_trainer,
                     all_g_data,
                     val_error):
        predictor_trainer.fit(all_g_data, val_error, logger=self.logger)

    def pred_acc_agent(self, predictor_trainer,
                       all_g_data):
        return predictor_trainer.pred(all_g_data).cpu().numpy()

    def get_best_candidate_architectures(self, candidate_acc, reverse=False):
        if reverse:
            return np.argsort(candidate_acc)[::-1][:self.cfg.SEARCH_STRATEGY.K]
        else:
            return np.argsort(candidate_acc)[:self.cfg.SEARCH_STRATEGY.K]