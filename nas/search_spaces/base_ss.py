from abc import ABCMeta, abstractmethod


class BaseSearchSpace(metaclass=ABCMeta):
    """
    This is the base class of search space and it provides the interfaces of
    the search space classes. All the cell based search space have to inherit from this
    base class.
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def __str__(self):
        pass

    @abstractmethod
    def random_architectures(self, num):
        """
        This method randomly generate a batch of architectures.
        """
        pass

    @abstractmethod
    def get_candidates(self, data, num, num_best_arches, mutate_rate=1.0, patience_factor=5.0):
        """
        This method generate architectures following a specific rule. Candidate architectures
        are generated by sequentially mutating the architectures in the architecture pool. Architecture
        with high performance have the priority to mutate first.
        """
        pass

    @abstractmethod
    def mutate_architecture(self, arch, mutate_rate=1.0, require_distance=False):
        """
        mutate a specific architecture
        """
        pass

    @abstractmethod
    def eval_architectures(self, arch_list):
        """
        evaluate the sampled neural architectures
        """
        pass

    def get_architectures_from_arch_list(self, arch_list: list):
        """
        get the neural architectures from the
        """
        pass

    @abstractmethod
    def init_search_space(self):
        pass

    @abstractmethod
    def query_arch(self, *args, **kvargs):
        pass

    def get_arch_acc_from_arch_list(self, arch_list: list):
        return [arch.val_error for arch in arch_list]