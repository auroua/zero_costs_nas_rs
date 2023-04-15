from abc import abstractmethod, ABCMeta


class BaseStrategy(metaclass=ABCMeta):
    def __init__(self, cfg, search_space, device=None):
        self.cfg = cfg
        self.search_space = search_space
        self.device = device

    @abstractmethod
    def do_search(self):
        pass

    @abstractmethod
    def update_agent(self, *vargs, **kargs):
        pass

    @abstractmethod
    def pred_acc_agent(self, *vargs, **kargs):
        pass

    @abstractmethod
    def get_best_candidate_architectures(self, *vargs, **kargs):
        pass

    def compute_best_test_losses(self, data):
        """
        Given full data from a completed nas algorithm,
        output the test error of the arch with the best val error
        after every multiple of k
        """
        k = self.cfg.SEARCH_STRATEGY.K
        total_queries = self.cfg.SEARCH_STRATEGY.SEARCH_BUDGET
        results = []
        best_arch_key_val_list = []
        for query in range(k, total_queries + k, k):
            if self.cfg.SEARCH_SPACE.TYPE == "SEG101":
                best_arch = sorted(data[:query], key=lambda i: i.miou, reverse=True)[0]
                test_error = best_arch.miou
                best_arch_key_val_list.append((best_arch.arch_key, best_arch))
            else:
                best_arch = sorted(data[:query], key=lambda i: i.val_error)[0]
                test_error = best_arch.test_error
            results.append((query, test_error))
        if self.cfg.SEARCH_SPACE.TYPE == "SEG101":
            return results, best_arch_key_val_list
        else:
            return results