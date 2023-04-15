import copy

from ..base_ss import BaseSearchSpace
from .arch_seg_201 import ArchSeg201
from ..build import SPACE_REGISTRY
from nas.utils.comm import get_hashkey
import logging
import random
from itertools import combinations
from nas.eigen.trainer_architectures import ansyc_multiple_process_evaluation


@SPACE_REGISTRY.register()
class SEG201(BaseSearchSpace):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.logger = logging.getLogger(__name__)
        # self.total_arch_dict = self.init_search_space()
        # self.total_keys = list(self.total_arch_dict.keys())
        self.allow_isomorphisms = cfg.SEARCH_STRATEGY.ALLOW_ISOMORPHISMS
        self.encode_method = cfg.SEARCH_STRATEGY.ENCODE_METHOD
        self.args = None
        self.cfg = cfg

        self.spatial_ops = self.cfg.SEARCH_SPACE.SEG_101.SPATIAL_OPS
        self.spatial_ops_s4 = self.cfg.SEARCH_SPACE.SEG_101.SPATIAL_OPS_S4
        self.channel_ops = self.cfg.SEARCH_SPACE.SEG_101.CHANNEL_OPS
        self.stage_ops = self.cfg.SEARCH_SPACE.SEG_101.STAGE_SEP_OPS
        self.c4_2 = list(combinations(range(4), 2))
        self.c3_2 = list(combinations(range(3), 2))
        self.head_merge_ops = self.cfg.SEARCH_SPACE.SEG_101.HEAD_MERGE_OPS

        self.total_ops = ["input"]
        self.total_ops.extend(self.spatial_ops)
        self.total_ops.extend(self.spatial_ops_s4)
        self.total_ops.extend(self.channel_ops)
        self.total_ops.extend(self.stage_ops)
        self.total_ops.append("FeatureMerge")
        self.total_ops.extend(self.head_merge_ops)

        self.total_ops = list(set(self.total_ops))

        self.mutate_part = ['spatial_ops', 'spatial_ops_s4', 'channel_ops', 'stage_ops', 'head_0_merge_idx_stage_1',
                            "head_0_merge_idx_stage_2", "head_1_merge_idx_stage_1", "head_1_merge_idx_stage_2",
                            "head_merge_ops"]
        self.logger.info(f"There are totally {self.init_search_space()} architectures in search space "
                         f"{self.cfg.SEARCH_SPACE.TYPE}.")

    def random_architectures(self, num):
        arch_dict = {}
        arch_list = []
        while True:
            spatial_ops = random.choices(self.spatial_ops, k=4)
            spatial_ops_s4 = random.choices(self.spatial_ops_s4, k=4)
            channel_ops = random.choice(self.channel_ops)
            stage_ops = random.choices(self.stage_ops, k=4)
            head_0_merge_idx_stage_1 = random.sample(self.c4_2, k=3)
            head_0_merge_idx_stage_2 = random.sample(self.c3_2, k=2)
            head_1_merge_idx_stage_1 = random.sample(self.c4_2, k=3)
            head_1_merge_idx_stage_2 = random.sample(self.c3_2, k=2)
            head_merge_op = random.choice(self.head_merge_ops)

            if all(op == "Zero" for op in stage_ops):
                continue

            arch_key = get_hashkey('_'.join([head_merge_op,
                                             *[str(head_0_merge_idx_stage_1[0]), str(head_0_merge_idx_stage_1[1]),
                                               str(head_0_merge_idx_stage_1[2])],
                                             *[str(head_0_merge_idx_stage_2[0]), str(head_0_merge_idx_stage_2[1])],
                                             *[str(head_1_merge_idx_stage_1[0]), str(head_1_merge_idx_stage_1[1]),
                                               str(head_1_merge_idx_stage_1[2])],
                                             *[str(head_1_merge_idx_stage_2[0]), str(head_1_merge_idx_stage_2[1])],
                                             *spatial_ops, *spatial_ops_s4, channel_ops,
                                             *stage_ops]))
            if arch_key in arch_dict:
                continue
            else:
                arch = ArchSeg201(
                    stages=self.cfg.SEARCH_SPACE.SEG_101.BACKBONE_STAGES,
                    groups=self.cfg.SEARCH_SPACE.SEG_101.GROUP_OPTION,
                    head_merge_ops=head_merge_op,
                    head_0_merge_idx_stage_1=head_0_merge_idx_stage_1,
                    head_0_merge_idx_stage_2=head_0_merge_idx_stage_2,
                    head_1_merge_idx_stage_1=head_1_merge_idx_stage_1,
                    head_1_merge_idx_stage_2=head_1_merge_idx_stage_2,
                    stage_op_lists=stage_ops,
                    channel_ops=channel_ops,
                    spatial_ops=spatial_ops,
                    spatial_ops_s4=spatial_ops_s4,
                    stage_merge_ratio=self.cfg.SEARCH_SPACE.SEG_101.STAGE_MERGE_OUT_RATIO,
                    cfg=self.cfg,
                    arch_key=arch_key,
                    total_ss_ops=self.total_ops
                )
                arch_dict[arch_key] = arch
                arch_list.append(arch)
                if len(arch_dict) >= num:
                    break
        return arch_list

    def get_candidates(self, data, num, num_best_arches, mutate_rate=1.0,
                       patience_factor=5, return_dist=False):
        candidate_list = []
        model_keys = [arch.arch_key for arch in data]
        best_arches = [arch for arch in sorted(data, key=lambda i: i.miou, reverse=True)[:num_best_arches * patience_factor]]
        for arch in best_arches:
            if len(candidate_list) >= num:
                break
            for i in range(num):
                mutated_arch = copy.deepcopy(arch)
                mutated_arch.iou_bg = 0
                mutated_arch.iou_target = 0
                mutated_arch.miou = 0
                mutated_arch.eval_results = None
                mutated_arch.mutate(base_mutate_rate=0.6,
                                    mutate_rate=mutate_rate,
                                    all_spatial_ops=self.spatial_ops,
                                    all_spatial_ops_s4=self.spatial_ops_s4,
                                    all_channel_ops=self.channel_ops,
                                    all_stage_ops=self.stage_ops,
                                    all_head_merge_ops=self.head_merge_ops,
                                    c4_2=self.c4_2,
                                    c3_2=self.c3_2)
                arch_k = mutated_arch.get_arch_key()
                mutated_arch.arch_key = arch_k
                if not arch_k in model_keys:
                    candidate_list.append(mutated_arch)
        return candidate_list

    def mutate_architecture(self, arch, mutate_rate=1.0, require_distance=False):
        pass

    def eval_architectures(self, arch_list):
        if not self.args:
            raise ValueError("The property args should not be None.")
        ansyc_multiple_process_evaluation(self.args, self.cfg, arch_list)
        for arch in arch_list:
            self.logger.info(f"=== Architecture Key: {arch.arch_key}, {arch.eval_results} ===")
        return arch_list

    def get_architectures_from_arch_list(self, arch_list: list):
        pass

    def get_arch_acc_from_arch_list(self, arch_list: list):
        return [arch.miou for arch in arch_list]

    def query_arch(self,
                   matrix,
                   ops):
        pass

    def init_search_space(self):
        """

        Returns: The count of architectures in this search space.
        """
        total_spatial_choices = len(self.spatial_ops)*len(self.spatial_ops)*len(self.spatial_ops)*len(self.spatial_ops)
        total_spatial_choices_s4 = len(self.spatial_ops_s4)*len(self.spatial_ops_s4)* \
                                   len(self.spatial_ops_s4)*len(self.spatial_ops_s4)
        total_channel_choices = len(self.channel_ops)
        total_stage_choices = len(self.stage_ops)*len(self.stage_ops)*len(self.stage_ops)*len(self.stage_ops)
        total_stage_merge_choices = 5*4*3 * 5*4*3
        head_merge_op = len(self.head_merge_ops)

        return total_spatial_choices * total_spatial_choices_s4 * total_channel_choices * total_stage_choices * \
               total_stage_merge_choices * head_merge_op
