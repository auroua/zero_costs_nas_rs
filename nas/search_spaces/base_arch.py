from abc import ABCMeta, abstractmethod


class Arch(metaclass=ABCMeta):
    def __init__(self, arch_matrix, arch_ops, isolate_node, arch_o_matrix, arch_o_ops, val_error, test_error, key=None,
                 encode_path=None, encode_adj=None, encode_path_aware=None, edge_idx=None, node_f=None, g_data=None,
                 search_space=None, cfg=None):
        self.arch_matrix = arch_matrix
        self.arch_ops = arch_ops
        self.isolate_node = isolate_node
        # original matrix
        self.arch_o_matrix = arch_o_matrix
        # original operations
        self.arch_o_ops = arch_o_ops
        # validation error
        self.val_error = val_error
        # test error
        self.test_error = test_error
        self.key = key
        self.encode_path_vec = encode_path
        self.encode_adj_vec = encode_adj
        self.encode_path_aware_vec = encode_path_aware
        self.edge_idx_vec = edge_idx
        self.node_f_vec = node_f
        self.g_arch = g_data
        self.search_space = search_space
        self.cfg = cfg

    @abstractmethod
    def assemble_neural_network(self):
        """
        build neural network from the adjacency matrix and operations
        Returns: neural network in search space
        """
        pass

    @abstractmethod
    def encode_path(self):
        """
        get the path based encoding of the given neural network architecture. i.e. adjacency matrix and operations
        Returns: the path-based encoding of input neural architecture.
        """
        pass

    @abstractmethod
    def encode_adj(self):
        """
        get the adjacency encoding of the given neural network architecture
        Returns: the adjacency encoding of the input neural network
        """
        pass

    @abstractmethod
    def encode_path_aware(self):
        """
        get the position-aware path-based encoding of the input neural network architecture
        Returns: the position-aware path-based encoding

        """
        pass

    @abstractmethod
    def edge_idx(self):
        """
        Returns: edge index of input neural network
        """
        pass

    @abstractmethod
    def node_f(self):
        """
        Returns: node features of the input neural network
        """
        pass

    @abstractmethod
    def g_data(self):
        """
        Returns: the graph data of input neural network
        """
        pass

    @abstractmethod
    def encode_graph(self, is_idx: bool = False, node_vec_len: int = 0):
        pass

    @abstractmethod
    def mutate(self, *args, **kargs):
        pass

    @abstractmethod
    def get_val_loss(self, nasbench, deterministic, patience=50):
        pass

    @abstractmethod
    def get_test_loss(self, nasbench, deterministic, patience=50):
        pass

    @abstractmethod
    def get_adj_dist(self, arch_2):
        pass

    def path_encoding(self):
        if self.cfg.SEARCH_STRATEGY.ENCODE_METHOD == "encode_path":
            self.encode_path_vec = self.encode_path()
        elif self.cfg.SEARCH_STRATEGY.ENCODE_METHOD == "encode_adj":
            self.encode_adj_vec = self.encode_adj()
        elif self.cfg.SEARCH_STRATEGY.ENCODE_METHOD == "encode_path_aware":
            self.encode_path_aware_vec = self.encode_path_aware()
        else:
            raise ValueError(f"Path encoding type {self.cfg.SEARCH_STRATEGY.ENCODE_METHOD} "
                             f"does not support at present!")

    def get_paths(self):
        """
        return all paths from input to output
        """
        num_vertices = self.cfg.SEARCH_SPACE.NASBENCH_101.NUM_VERTICES
        paths = []
        for j in range(0, num_vertices):
            paths.append([[]]) if self.arch_matrix[0][j] else paths.append([])

        # create paths sequentially
        for i in range(1, num_vertices - 1):
            for j in range(1, num_vertices):
                if self.arch_matrix[i][j]:
                    for path in paths[i]:
                        paths[j].append([*path, self.arch_ops[i]])
        return paths[-1]

    def get_path_indices(self):
        """
        compute the index of each path
        There are 3^0 + ... + 3^5 paths total.
        (Paths can be length 0 to 5, and for each path, for each node, there
        are three choices for the operation.)
        """
        paths = self.get_paths()
        OPS = self.cfg.SEARCH_SPACE.NASBENCH_101.OPS[1:-2]
        mapping = {v: idx for idx, v in enumerate(OPS)}
        path_indices = []
        for path in paths:
            index = 0
            for i in range(self.cfg.SEARCH_SPACE.NASBENCH_101.NUM_VERTICES - 1):
                if i == len(path):
                    path_indices.append(index)
                    break
                else:
                    index += len(OPS) ** i * (mapping[path[i]] + 1)
        return tuple(path_indices)