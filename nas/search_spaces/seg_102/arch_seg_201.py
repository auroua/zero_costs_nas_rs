from nas.search_spaces.seg_101.assembel_seg_101 import assemble_architecture_seg_101
import numpy as np
import torch
from torch_geometric.data import Data
from nas.utils.comm import get_hashkey
import random


class ArchSeg201:
    def __init__(self,
                 stages,
                 groups,
                 head_merge_ops,
                 head_0_merge_idx_stage_1,
                 head_0_merge_idx_stage_2,
                 head_1_merge_idx_stage_1,
                 head_1_merge_idx_stage_2,
                 stage_op_lists,
                 channel_ops,
                 spatial_ops,
                 spatial_ops_s4,
                 stage_merge_ratio,
                 cfg,
                 arch_key,
                 total_ss_ops):
        self.stages = stages
        self.groups = groups
        self.head_merge_ops = head_merge_ops
        self.head_0_merge_idx_stage_1 = head_0_merge_idx_stage_1
        self.head_0_merge_idx_stage_2 = head_0_merge_idx_stage_2
        self.head_1_merge_idx_stage_1 = head_1_merge_idx_stage_1
        self.head_1_merge_idx_stage_2 = head_1_merge_idx_stage_2
        self.stage_op_lists = stage_op_lists
        self.channel_ops = channel_ops
        self.spatial_ops = spatial_ops
        self.spatial_ops_s4 = spatial_ops_s4
        self.stage_merge_ratio = stage_merge_ratio
        self.cfg = cfg
        self.arch_key = arch_key

        self.miou = 0.0
        self.iou_bg = 0.0
        self.iou_target = 0.0

        self.eval_results = None
        self.total_ss_ops = total_ss_ops

    def encode_path(self):
        pass

    def encode_adj(self):
        pass

    def encode_path_aware(self):
        pass

    def edge_idx(self):
        pass

    def node_f(self):
        pass

    def g_data(self):
        adj_matrix, ops = self.get_adj_matrix_and_ops()
        edge_idx, node_feature = self.graph_ops_to_graph((adj_matrix, ops))
        graph_data = Data(edge_index=edge_idx.long(), x=node_feature.float())
        return graph_data

    def assemble_neural_network(self, cfg_seg):
        kvargs = {
            'stages': self.stages,
            'groups': self.groups,
            'stage_op_lists': self.stage_op_lists,
            'head_0_merge_idx_stage_1': self.head_0_merge_idx_stage_1,
            'head_0_merge_idx_stage_2': self.head_0_merge_idx_stage_2,
            'head_1_merge_idx_stage_1': self.head_1_merge_idx_stage_1,
            'head_1_merge_idx_stage_2': self.head_1_merge_idx_stage_2,
            'channel_ops': self.channel_ops,
            'head_merge_ops': self.head_merge_ops,
            'spatial_ops': self.spatial_ops,
            'spatial_ops_s4': self.spatial_ops_s4,
            'stage_merge_ratio': self.stage_merge_ratio,
            'loss_func': cfg_seg.MODEL.SEM_SEG_HEAD.LOSS_TYPE,
            'num_classes': cfg_seg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            'loss_weight': cfg_seg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        }
        return assemble_architecture_seg_101(kvargs, cfg_seg)

    def encode_graph(self, is_idx: bool = False, node_vec_len: int = 0):
        pass

    def mutate(self, base_mutate_rate, mutate_rate, all_spatial_ops, all_spatial_ops_s4, all_channel_ops,
               all_stage_ops, all_head_merge_ops, c4_2, c3_2):
        group_mutate_rate = 1.5 * mutate_rate / self.groups
        # stage s1 and s2 mutation
        if random.random() < base_mutate_rate:
            # Spation Group Operation mutation
            for i in range(len(self.spatial_ops)):
                if random.random() < group_mutate_rate:
                    while True:
                        sp_op = random.sample(all_spatial_ops, 1)[0]
                        if self.spatial_ops[i] != sp_op:
                            self.spatial_ops[i] = sp_op
                            break
                        else:
                            continue

        if random.random() < base_mutate_rate:
            # Spation Group Operation mutation
            for i in range(len(self.spatial_ops_s4)):
                if random.random() < group_mutate_rate:
                    while True:
                        sp_op = random.sample(all_spatial_ops_s4, 1)[0]
                        if self.spatial_ops_s4[i] != sp_op:
                            self.spatial_ops_s4[i] = sp_op
                            break
                        else:
                            continue

        # channel attention mutation rate
        if random.random() < base_mutate_rate:
            if len(all_channel_ops) > 1:
                while True:
                    ch_op = random.sample(all_channel_ops, 1)[0]
                    if ch_op != self.channel_ops:
                        self.channel_ops = ch_op
                        break
                    else:
                        continue

        # stage operation mutation rate
        stage_op_mutation_rate = 1.0 / len(self.stages)
        if random.random() < base_mutate_rate:
            for i in range(len(self.stage_op_lists)):
                if random.random() < stage_op_mutation_rate:
                    op = random.sample(all_stage_ops, 1)[0]
                    if self.stage_op_lists[i] != op:
                        self.stage_op_lists[i] = op

        if random.random() < base_mutate_rate:
            while True:
                merge_op = random.sample(all_head_merge_ops, 1)[0]
                if not self.head_merge_ops == merge_op:
                    self.head_merge_ops = merge_op
                    break
                else:
                    continue

        # head 0 stage 1 mutate
        if random.random() < base_mutate_rate:
            self.head_0_merge_idx_stage_1 = random.sample(c4_2, k=3)

        # head 1 stage 1 mutate
        if random.random() < base_mutate_rate:
            self.head_1_merge_idx_stage_1 = random.sample(c4_2, k=3)

        # head 0 stage 2 mutate
        if random.random() < base_mutate_rate:
            self.head_0_merge_idx_stage_2 = random.sample(c3_2, k=2)

        # head 1 stage 2 mutate
        if random.random() < base_mutate_rate:
            self.head_1_merge_idx_stage_2 = random.sample(c3_2, k=2)

    def get_val_loss(self):
        return self.miou

    def get_adj_dist(self):
        return self.miou

    def get_arch_key(self):
        arch_key = get_hashkey('_'.join([self.head_merge_ops,
                                         *[str(self.head_0_merge_idx_stage_1[0]), str(self.head_0_merge_idx_stage_1[1]),
                                           str(self.head_0_merge_idx_stage_1[2])],
                                         *[str(self.head_0_merge_idx_stage_2[0]), str(self.head_0_merge_idx_stage_2[1])],
                                         *[str(self.head_1_merge_idx_stage_1[0]), str(self.head_1_merge_idx_stage_1[1]),
                                           str(self.head_1_merge_idx_stage_1[2])],
                                         *[str(self.head_1_merge_idx_stage_2[0]), str(self.head_1_merge_idx_stage_2[1])],
                                         *self.spatial_ops, *self.spatial_ops_s4, self.channel_ops,
                                         *self.stage_op_lists]))
        return arch_key

    def __str__(self):
        spatial_operations = "[ " + "  ".join(self.spatial_ops) + " ]"
        spatial_operations_s4 = "[ " + "  ".join(self.spatial_ops_s4) + " ]"
        stage_op_list_str = "[ " + "  ".join(self.stage_op_lists) + " ]"
        head_0_merge_0 = "[" + "  ".join(map(str, self.head_0_merge_idx_stage_1)) + "]"
        head_0_merge_1 = "[" + "  ".join(map(str, self.head_0_merge_idx_stage_2)) + "]"
        head_1_merge_0 = "[" + "  ".join(map(str, self.head_1_merge_idx_stage_1)) + "]"
        head_1_merge_1 = "[" + "  ".join(map(str, self.head_1_merge_idx_stage_2)) + "]"

        return f"Spatial Operations: {spatial_operations}, " \
               f"Spatial Operations S4: {spatial_operations_s4}, " \
               f"Channel Attentions: {self.channel_ops}, " \
               f"Stage Operations: {stage_op_list_str}, " \
               f"Head 0 Merge Stage 0: {head_0_merge_0}, " \
               f"Head 0 Merge Stage 1: {head_0_merge_1}, " \
               f"Head 1 Merge Stage 0: {head_1_merge_0}, " \
               f"Head 1 Merge Stage 1: {head_1_merge_1}, " \
               f"Head Merge Operations: {self.head_merge_ops}. "

    def get_adj_matrix_and_ops(self):
        nodes_num = (self.groups + 1) * len(self.stages) + len(self.stages) * 2 + \
                    len(self.head_0_merge_idx_stage_1) + len(self.head_0_merge_idx_stage_2) + \
                    len(self.head_1_merge_idx_stage_1) + len(self.head_1_merge_idx_stage_2) + 1 + 1 + 1

        adj_matrix = np.zeros((nodes_num, nodes_num), dtype=np.int8)
        total_ops = []

        # stage s1
        adj_matrix[0, 20] = 1
        adj_matrix[1, 20] = 1
        adj_matrix[2, 20] = 1
        adj_matrix[3, 20] = 1
        adj_matrix[4, 20] = 1
        adj_matrix[20, 24] = 1

        # stage s2
        adj_matrix[5, 21] = 1
        adj_matrix[6, 21] = 1
        adj_matrix[7, 21] = 1
        adj_matrix[8, 21] = 1
        adj_matrix[9, 21] = 1
        adj_matrix[21, 25] = 1

        # stage s3
        adj_matrix[10, 22] = 1
        adj_matrix[11, 22] = 1
        adj_matrix[12, 22] = 1
        adj_matrix[13, 22] = 1
        adj_matrix[14, 22] = 1
        adj_matrix[22, 26] = 1

        # stage s4
        adj_matrix[15, 23] = 1
        adj_matrix[16, 23] = 1
        adj_matrix[17, 23] = 1
        adj_matrix[18, 23] = 1
        adj_matrix[19, 23] = 1
        adj_matrix[23, 27] = 1

        for idx_out, pair in enumerate(self.head_0_merge_idx_stage_1):
            idx_1, idx_2 = pair
            adj_matrix[24+idx_1, 28+idx_out] = 1
            adj_matrix[24+idx_2, 28+idx_out] = 1

        for idx_out, pair in enumerate(self.head_1_merge_idx_stage_1):
            idx_1, idx_2 = pair
            adj_matrix[24+idx_1, 31+idx_out] = 1
            adj_matrix[24+idx_2, 31+idx_out] = 1

        for idx_out, pair in enumerate(self.head_0_merge_idx_stage_2):
            idx_1, idx_2 = pair
            adj_matrix[28+idx_1, 34+idx_out] = 1
            adj_matrix[28+idx_2, 34+idx_out] = 1

        for idx_out, pair in enumerate(self.head_1_merge_idx_stage_2):
            idx_1, idx_2 = pair
            adj_matrix[31+idx_1, 36+idx_out] = 1
            adj_matrix[31+idx_2, 36+idx_out] = 1

        # head 0 merge
        adj_matrix[34, 38] = 1
        adj_matrix[35, 38] = 1

        # head 1 merge
        adj_matrix[36, 39] = 1
        adj_matrix[37, 39] = 1

        # head merge
        adj_matrix[38, 40] = 1
        adj_matrix[39, 40] = 1

        # stage s1
        total_ops.append("input")
        for op in self.spatial_ops:
            total_ops.append(op)

        # stage s2
        total_ops.append("input")
        for op in self.spatial_ops:
            total_ops.append(op)

        # stage s3
        total_ops.append("input")
        for op in self.spatial_ops_s4:
            total_ops.append(op)

        # stage s4
        total_ops.append("input")
        for op in self.spatial_ops_s4:
            total_ops.append(op)

        for _ in self.stages:
            total_ops.append(self.channel_ops)

        for stage_op in self.stage_op_lists:
            total_ops.append(stage_op)

        # head 0 merge stage 1
        total_ops.append("FeatureMerge")
        total_ops.append("FeatureMerge")
        total_ops.append("FeatureMerge")

        # head 1 merge stage 1
        total_ops.append("FeatureMerge")
        total_ops.append("FeatureMerge")
        total_ops.append("FeatureMerge")

        # head 0 merge stage 2
        total_ops.append("FeatureMerge")
        total_ops.append("FeatureMerge")

        # head 1 merge stage 2
        total_ops.append("FeatureMerge")
        total_ops.append("FeatureMerge")
        # head 0 merge
        total_ops.append("FeatureMerge")
        # head 1 merge
        total_ops.append("FeatureMerge")
        # head 0 and head 1 merge
        total_ops.append(self.head_merge_ops)

        return adj_matrix, total_ops

    def graph_ops_to_graph(self, graph_data):
        matrix, ops = graph_data[0], graph_data[1]
        total_ops = self.total_ss_ops
        NUM_VERTICES = matrix.shape[0]
        node_feature = torch.zeros(matrix.shape[0], len(total_ops))
        edges = int(np.sum(matrix))
        edge_idx = torch.zeros(2, edges)
        counter = 0
        for i in range(NUM_VERTICES):
            idx = total_ops.index(ops[i])
            node_feature[i, idx] = 1
            for j in range(NUM_VERTICES):
                if matrix[i, j] == 1:
                    edge_idx[0, counter] = i
                    edge_idx[1, counter] = j
                    counter += 1
        return edge_idx, node_feature