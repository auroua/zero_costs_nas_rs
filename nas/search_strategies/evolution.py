from .base import BaseStrategy
import logging
from nas.search_strategies.build import STRATEGY_REGISTRY
import numpy as np
import random


@STRATEGY_REGISTRY.register()
class Evolutionary(BaseStrategy):
    def __init__(self, cfg, search_space, device):
        BaseStrategy.__init__(self, cfg, search_space, device)
        self.logger = logging.getLogger(f"nas_gpu_{device}")

    def do_search(self):
        data = self.search_space.random_architectures(num=self.cfg.SEARCH_STRATEGY.NUM_INIT)

        query = self.cfg.SEARCH_STRATEGY.NUM_INIT + self.cfg.SEARCH_STRATEGY.K
        if self.cfg.SEARCH_SPACE.HAS_GT == "N":
            data = self.search_space.eval_architectures(data)
        val_losses = self.search_space.get_arch_acc_from_arch_list(data)
        if self.cfg.SEARCH_STRATEGY.NUM_INIT <= self.cfg.SEARCH_STRATEGY.EVOLUTIONARY.POPULATION_SIZE:
            population = [i for i in range(self.cfg.SEARCH_STRATEGY.NUM_INIT)]
        else:
            population = np.argsort(val_losses)[:self.cfg.SEARCH_STRATEGY.EVOLUTIONARY.POPULATION_SIZE].tolist()

        while query <= self.cfg.SEARCH_STRATEGY.SEARCH_BUDGET:
            sample = random.sample(population, self.cfg.SEARCH_STRATEGY.EVOLUTIONARY.TOURNAMENT_SIZE)
            best_index = sorted([(i, val_losses[i]) for i in sample], key=lambda i: i[1])[0][0]
            # best_arch_dict = {'matrix': data[best_index].arch_matrix, 'ops': data[best_index].arch_ops,
            #                   'key': data[best_index].key}
            mutated = self.search_space.mutate_architecture(data[best_index],
                                                            self.cfg.SEARCH_STRATEGY.EVOLUTIONARY.MUTATION_RATE)
            if self.cfg.SEARCH_SPACE.HAS_GT == "N":
                mutated = self.search_space.eval_architectures(mutated)[0]
                val_losses.append(mutated.miou)
            else:
                val_losses.append(mutated.val_error)
            data.append(mutated)
            population.append(len(data) - 1)
            # kill the worst from the population   in nas bench paper kill the oldest arch
            if len(population) > self.cfg.SEARCH_STRATEGY.EVOLUTIONARY.POPULATION_SIZE:
                worst_index = sorted([(i, val_losses[i]) for i in population], key=lambda i: i[1])[-1][0]
                population.remove(worst_index)

            if query % 10 == 0 and query != 0:
                top_5_loss = sorted(self.search_space.get_arch_acc_from_arch_list(data))[:min(5, len(data))]
                self.logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
            query += 1
        return data

    def update_agent(self):
        pass

    def pred_acc_agent(self, *vargs, **kargs):
        pass

    def get_best_candidate_architectures(self, *vargs, **kargs):
        pass