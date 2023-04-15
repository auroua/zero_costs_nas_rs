from ..base_ss import BaseSearchSpace
from .arch_nasbench_101 import ArchNasBench101
from ..build import SPACE_REGISTRY
from nasbench import api
import numpy as np
from nas.utils.search_space import find_isolate_node
import random


@SPACE_REGISTRY.register()
class NasBench101(BaseSearchSpace):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ops_all = cfg.SEARCH_SPACE.NASBENCH_101.OPS
        self.ops_dict = {idx: op for idx, op in enumerate(self.ops_all)}
        self.ops_core = self.ops_all[1:-2]
        self.num_vertices = cfg.SEARCH_SPACE.NASBENCH_101.NUM_VERTICES
        self.max_edges = cfg.SEARCH_SPACE.NASBENCH_101.MAX_EDGES
        self.graph_node_dim = cfg.SEARCH_SPACE.NASBENCH_101.GRAPH_NODE_DIM
        self.seq_aware_path_len = cfg.SEARCH_SPACE.NASBENCH_101.SEQ_AWARE_PATH_LEN
        self.nasbench = api.NASBench(cfg.SEARCH_SPACE.NASBENCH_101.TF_FILE_PATH)
        self.total_archs, self.total_keys = self.init_search_space()

        self.allow_isomorphisms = cfg.SEARCH_STRATEGY.ALLOW_ISOMORPHISMS
        self.encode_method = cfg.SEARCH_STRATEGY.ENCODE_METHOD

    def random_architectures(self, num):
        data = []
        dic = {}
        key_list = []
        while True:
            k = random.sample(self.total_keys, 1)
            key_list.append(k)
            arch = self.total_archs[k[0]]
            arch.path_encoding()
            path_indices = arch.get_path_indices()
            if self.allow_isomorphisms or path_indices not in dic:
                dic[path_indices] = 1
                data.append(arch)
            if len(data) == num:
                break
        return data

    def get_candidates(self, data, num, num_best_arches, mutate_rate=1.0,
                       patience_factor=5, return_dist=False):
        candidates = []
        dic = {}
        dist_list = []
        nums_list = []
        mutated_archs_list = []
        for d in data:
            path_indices = d.get_path_indices()
            dic[path_indices] = 1

        mutate_arch_dict = {}
        # mutate architectures with the lowest validation error
        best_arches = [arch for arch in sorted(data, key=lambda i: i.val_error)[:num_best_arches * patience_factor]]
        best_arch_datas = [d for d in sorted(data, key=lambda i: i.val_error)[:num_best_arches * patience_factor]]
        # stop when candidates is size num
        # use patience_factor instead of a while loop to avoid long or infinite runtime
        for idx, arch in enumerate(best_arches):
            if len(candidates) >= num:
                break
            nums = 0
            mutate_arch_dict[idx] = 0
            for i in range(num):
                mutated_arch = self.mutate_architecture(arch,
                                                        mutate_rate=mutate_rate)
                path_indices = mutated_arch.get_path_indices()
                if self.allow_isomorphisms or path_indices not in dic:
                    dic[path_indices] = 1
                    candidates.append(mutated_arch)
                    mutate_arch_dict[idx] += 1
                    dist = arch.get_adj_dist(mutated_arch)
                    dist_list.append(dist)
                    nums += 1
            nums_list.append(nums)
            mutated_archs_list.append(best_arch_datas[idx])
        if return_dist:
            return candidates[:num], dist_list[:num], 0, nums_list, mutated_archs_list
        else:
            return candidates[:num]

    def mutate_architecture(self, arch, mutate_rate=1.0, require_distance=False):
        while True:
            arch_mutate = arch.mutate(self.nasbench, mutate_rate)
            results = self.query_arch(matrix=arch_mutate["matrix"],
                                      ops=arch_mutate["ops"])
            if results:
                break
        return results

    def query_arch(self,
                   matrix,
                   ops):
        matrix = matrix.astype(np.int8)
        model_spec = api.ModelSpec(matrix=matrix, ops=ops)
        key = model_spec.hash_spec(self.ops_core)
        if not model_spec.valid_spec:
            return None

        o_matrix = model_spec.matrix
        o_ops = model_spec.ops

        if key in self.total_keys:
            arch = self.total_archs[key]
        else:
            matrix, ops = self.matrix_dummy_nodes(o_matrix, o_ops)
            arch = ArchNasBench101(arch_matrix=matrix, arch_ops=ops,
                                   arch_o_matrix=o_matrix, arch_o_ops=o_ops, cfg=self.cfg)
            arch.val_error = arch.get_val_loss(self.nasbench, deterministic=True)
            arch.test_error = arch.get_test_loss(self.nasbench, deterministic=True)
            arch.key = key
            arch.path_encoding()
        return arch

    def eval_architectures(self, arch_list: list):
        return arch_list

    def init_search_space(self):
        total_arch = {}
        total_keys = [k for k in self.nasbench.computed_statistics]

        best_key = None
        best_val = 0
        for k in total_keys:
            val_acc = []
            test_acc = []
            arch_matrix = self.nasbench.fixed_statistics[k]['module_adjacency']
            arch_ops = self.nasbench.fixed_statistics[k]['module_operations']
            if arch_matrix.shape[0] < 7:
                matrix, ops = self._matrix_dummy_nodes(arch_matrix, arch_ops)
            else:
                matrix = arch_matrix
                ops = arch_ops
            spec = api.ModelSpec(matrix=arch_matrix, ops=arch_ops)
            isolate_list = []
            if arch_matrix.shape[0] == 7:
                isolate_list = find_isolate_node(arch_matrix)
                if len(isolate_list) >= 1:
                    print(arch_matrix)
                    print(isolate_list)
            if not self.nasbench.is_valid(spec):
                continue
            for i in range(3):
                val_acc.append(self.nasbench.computed_statistics[k][108][i]['final_validation_accuracy'])
                test_acc.append(self.nasbench.computed_statistics[k][108][i]['final_test_accuracy'])
            val_mean = float(np.mean(val_acc))
            test_mean = float(np.mean(test_acc))

            if best_val < val_mean:
                best_val = val_mean
                best_key = k

            total_arch[k] = ArchNasBench101(
                arch_matrix=matrix,
                arch_ops=ops,
                arch_o_matrix=arch_matrix,
                arch_o_ops=arch_ops,
                isolate_node=isolate_list,
                val_error=100*(1-val_mean),
                test_error=100*(1-test_mean),
                key=k,
                cfg=self.cfg
            )

        best_arch = total_arch[best_key]
        print(best_arch.val_error, best_arch.test_error)
        return total_arch, total_keys

    def _matrix_dummy_nodes(self, matrix_in, ops_in):
        # {2, 3, 4, 5, 6, 7}
        matrix = np.zeros((self.num_vertices, self.num_vertices))
        for i in range(matrix_in.shape[0]):
            idxs = np.where(matrix_in[i] == 1)
            for id in idxs[0]:
                if id == matrix_in.shape[0] - 1:
                    matrix[i, 6] = 1
                else:
                    matrix[i, id] = 1
        ops = ops_in[:(matrix_in.shape[0]-1)] + ['isolate'] * (7-matrix_in.shape[0]) + ops_in[-1:]
        find_isolate_node(matrix)
        return matrix, ops