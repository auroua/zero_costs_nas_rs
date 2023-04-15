import numpy as np
from ..base_arch import Arch
import copy
import random
from nasbench import api
from nas.utils.search_space import arch2edge_idx, arch2node_feature
from torch_geometric.data import Data


class ArchNasBench101(Arch):
    def __init__(self, arch_matrix,
                 arch_ops,
                 arch_o_matrix,
                 arch_o_ops,
                 isolate_node=None,
                 val_error=None,
                 test_error=None,
                 key=None,
                 encode_path=None,
                 encode_adj=None,
                 encode_path_aware=None,
                 edge_idx=None,
                 node_f=None,
                 g_data=None,
                 cfg=None):
        super().__init__(arch_matrix=arch_matrix, arch_ops=arch_ops, isolate_node=isolate_node,
                         arch_o_matrix=arch_o_matrix, arch_o_ops=arch_o_ops, val_error=val_error,
                         test_error=test_error, key=key, encode_path=encode_path, encode_adj=encode_adj,
                         encode_path_aware=encode_path_aware, edge_idx=edge_idx, node_f=node_f, g_data=g_data,
                         search_space="NasBench101", cfg=cfg)
        self.isolate_node = [] if not self.isolate_node else self.isolate_node
        self.num_vertices = self.cfg.SEARCH_SPACE.NASBENCH_101.NUM_VERTICES
        self.op_spots = self.cfg.SEARCH_SPACE.NASBENCH_101.NUM_VERTICES - 2
        self.ops = self.cfg.SEARCH_SPACE.NASBENCH_101.OPS
        self.ops_op = self.cfg.SEARCH_SPACE.NASBENCH_101.OPS[1:-2]

    def encode_path(self):
        """ output one-hot encoding of paths """
        num_paths = sum([len(self.ops_op) ** i for i in range(self.op_spots + 1)])
        path_indices = self.get_path_indices()
        path_encoding = np.zeros(num_paths)
        for index in path_indices:
            path_encoding[index] = 1
        return path_encoding

    def encode_adj(self):
        OPS = self.cfg.SEARCH_SPACE.NASBENCH_101.OPS[1:-2] + [self.cfg.SEARCH_SPACE.NASBENCH_101.OPS[-1]]
        encoding_length = (self.num_vertices ** 2 - self.num_vertices) // 2 + self.op_spots * len(OPS)
        encoding = np.zeros((encoding_length))
        n = 0
        for i in range(self.num_vertices - 1):
            for j in range(i + 1, self.num_vertices):
                encoding[n] = self.arch_matrix[i][j]
                n += 1
        for i in range(1, self.num_vertices - 1):
            op_idx = OPS.index(self.arch_ops[i])
            encoding[n + op_idx] = 1
            n += 4
        return tuple(encoding)

    def encode_path_aware(self):
        pass

    def edge_idx(self):
        if not self.edge_idx_vec:
            self.edge_idx_vec = arch2edge_idx(self.arch_matrix, self.num_vertices)
        return self.edge_idx_vec

    def node_f(self):
        if not self.node_f_vec:
            self.node_f_vec = arch2node_feature(self.arch_ops, self.num_vertices, len(self.ops), self.ops)
        return self.node_f_vec

    def g_data(self):
        if not self.g_arch:
            self.g_arch = Data(edge_index=self.edge_idx().long(), x=self.node_f().float())
        return self.g_arch

    def assemble_neural_network(self):
        pass

    def encode_graph(self, is_idx: bool = False, node_vec_len: int = 0):
        pass

    def mutate(self, nasbench, mutation_rate):
        iteration = 0
        while True:
            new_matrix = copy.deepcopy(self.arch_matrix)
            new_ops = copy.deepcopy(self.arch_ops)

            vertices = self.arch_matrix.shape[0]
            op_spots = vertices - 2
            edge_mutation_prob = mutation_rate / vertices
            for src in range(0, vertices - 1):
                for dst in range(src + 1, vertices):
                    if random.random() < edge_mutation_prob:
                        new_matrix[src, dst] = 1 - new_matrix[src, dst]

            if op_spots != 0:
                op_mutation_prob = mutation_rate / op_spots
                for ind in range(1, op_spots + 1):
                    if random.random() < op_mutation_prob:
                        available = [o for o in self.ops_op if o != new_ops[ind]]
                        new_ops[ind] = random.choice(available)

            new_spec = api.ModelSpec(new_matrix, new_ops)
            ops_idx = [-1] + [self.ops.index(new_ops[idx]) for idx in range(1, len(new_ops)-1)] + [-2]
            iteration += 1
            if iteration == 500:
                ops_idx = [-1] + [self.ops.index(self.arch_ops[idx]) for idx in range(1, len(self.arch_ops) - 1)] + [-2]
                return {
                    'matrix': copy.deepcopy(self.arch_matrix),
                    'ops': copy.deepcopy(self.arch_ops),
                    'ops_idx': ops_idx
                }
            if nasbench.is_valid(new_spec):
                return {
                    'matrix': new_matrix,
                    'ops': new_ops,
                    'ops_idx': ops_idx
                }

    def get_val_loss(self, nasbench, deterministic, patience=50):
        if not deterministic:
            # output one of the three validation accuracies at random
            return 100 * (1 - nasbench.query(api.ModelSpec(matrix=self.arch_matrix, ops=self.arch_ops))['validation_accuracy'])
        else:
            # query the api until we see all three accuracies, then average them
            # a few architectures only have two accuracies, so we use patience to avoid an infinite loop
            accs = []
            while len(accs) < 3 and patience > 0:
                patience -= 1
                acc = nasbench.query(api.ModelSpec(matrix=self.arch_matrix, ops=self.arch_ops))['validation_accuracy']
                if acc not in accs:
                    accs.append(acc)
            return round(100 * (1 - np.mean(accs)), 3)

    def get_test_loss(self, nasbench, deterministic, patience=50):
        """
        query the api until we see all three accuracies, then average them
        a few architectures only have two accuracies, so we use patience to avoid an infinite loop
        """
        accs = []
        while len(accs) < 3 and patience > 0:
            patience -= 1
            acc = nasbench.query(api.ModelSpec(matrix=self.arch_matrix, ops=self.arch_ops))['test_accuracy']
            if acc not in accs:
                accs.append(acc)
        return round(100 * (1 - np.mean(accs)), 3)

    def get_adj_dist(self, arch_2):
        """
        compute the distance between two architectures
        by comparing their adjacency matrices and op lists
        (edit distance)
        """

        graph_dist = np.sum(self.arch_matrix != arch_2.arch_matrix)
        ops_dist = np.sum(self.arch_ops != arch_2.arch_ops)
        return graph_dist + ops_dist