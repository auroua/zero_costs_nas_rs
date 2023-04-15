from nas.search_spaces.seg_101.seg_101_model import Seg101Model
from detectron2.utils.regnet_utils import REGNET_CHANNELS
from nas.layers.build import build_seg_nas_layers


def assemble_architecture_seg_101(arch_info, cfg_seg):
    stages = arch_info['stages']
    groups = arch_info['groups']
    spatial_ops = arch_info['spatial_ops']
    spatial_ops_s4 = arch_info['spatial_ops_s4']
    stage_channel_ops = arch_info['channel_ops']
    stage_op_lists = arch_info['stage_op_lists']
    head_0_merge_idx_stage_1 = arch_info['head_0_merge_idx_stage_1']
    head_0_merge_idx_stage_2 = arch_info['head_0_merge_idx_stage_2']
    head_1_merge_idx_stage_1 = arch_info['head_1_merge_idx_stage_1']
    head_1_merge_idx_stage_2 = arch_info['head_1_merge_idx_stage_2']
    stage_merge_ratio = arch_info['stage_merge_ratio']
    head_merge_ops = arch_info['head_merge_ops']

    num_classes = arch_info['num_classes']
    loss_weight = arch_info['loss_weight']

    input_size = cfg_seg.INPUT.CROP.SIZE[0]
    hw_ratio = get_img_hw_ratio(cfg_seg.DATASETS.TRAIN[0])
    input_size = [r*input_size for r in hw_ratio]
    regnet_channels = REGNET_CHANNELS[cfg_seg.MODEL.REGNETS.TYPE]

    stage_op_dict = build_stage_spatial_ops(
        stages=stages[:-2],
        groups=groups,
        spatial_ops=spatial_ops,
        channels=regnet_channels,
        input_img_size=input_size,
        stage_strides=cfg_seg.MODEL.REGNETS.STAGE_STRIDES
    )
    stage_op_dict_s4 = build_stage_spatial_ops(
        stages=stages[-2:],
        groups=groups,
        spatial_ops=spatial_ops_s4,
        channels=regnet_channels,
        input_img_size=input_size,
        stage_strides=cfg_seg.MODEL.REGNETS.STAGE_STRIDES
    )
    stage_op_dict[stages[-2]] = stage_op_dict_s4[stages[-2]]
    stage_op_dict[stages[-1]] = stage_op_dict_s4[stages[-1]]

    stage_group_head_dict = {}
    for stage_k, stage_op_v in stage_op_dict.items():
        stage_group_head_dict[f'{stage_k}_group'] = build_seg_nas_layers("StageGroup", groups=groups,
                                                                         group_conv_dict=stage_op_v)
    stage_channel_op_dict = build_channel_merge_ops(arch_op=stage_channel_ops,
                                                    channels=regnet_channels,
                                                    stages=stages,
                                                    input_img_size=input_size,
                                                    stage_strides=cfg_seg.MODEL.REGNETS.STAGE_STRIDES)
    stage_op_dict = build_stage_ops(stage_op_lists=stage_op_lists,
                                    channels=regnet_channels,
                                    stages=stages,
                                    input_img_size=input_size,
                                    stage_strides=cfg_seg.MODEL.REGNETS.STAGE_STRIDES)
    head_0_stage_0_dict, head_0_stage_1_dict, \
    head_1_stage_0_dict, head_1_stage_1_dict, \
    head_0_stage_0_keys, head_0_stage_1_keys, \
    head_1_stage_0_keys, head_1_stage_1_keys = build_stage_merge_ops(head_0_merge_idx_stage_1=head_0_merge_idx_stage_1,
                                                                     head_0_merge_idx_stage_2=head_0_merge_idx_stage_2,
                                                                     head_1_merge_idx_stage_1=head_1_merge_idx_stage_1,
                                                                     head_1_merge_idx_stage_2=head_1_merge_idx_stage_2,
                                                                     channels=regnet_channels,
                                                                     stages=stages,
                                                                     input_img_size=input_size,
                                                                     stage_strides=cfg_seg.MODEL.REGNETS.STAGE_STRIDES,
                                                                     stage_merge_ratio=stage_merge_ratio
                                                                     )

    head_channels_0 = [arch.get_sizes()["out_channels"] for arch in head_0_stage_1_dict.values()]
    head_spatial_0 = [arch.get_sizes()["out_spatial_size"] for arch in head_0_stage_1_dict.values()]

    head_0_merge = build_seg_nas_layers("FeatureMerge", in_channels=head_channels_0, out_channels=head_channels_0,
                                        spatial_sizes=head_spatial_0, name="head_0_merge")

    head_channels_1 = [arch.get_sizes()["out_channels"] for arch in head_1_stage_1_dict.values()]
    head_spatial_1 = [arch.get_sizes()["out_spatial_size"] for arch in head_1_stage_1_dict.values()]

    head_1_merge = build_seg_nas_layers("FeatureMerge", in_channels=head_channels_1, out_channels=head_channels_1,
                                        spatial_sizes=head_spatial_1, name="head_1_merge")

    head_merge = build_head_merge_ops(head_merge_ops, head_0_merge.get_sizes(), head_1_merge.get_sizes(),
                                      num_classes)

    return Seg101Model(
        stages=stages,
        groups=groups,
        stage_group_lists=stage_group_head_dict,
        stage_channel_dict=stage_channel_op_dict,
        stage_op_dict=stage_op_dict,
        head_0_stage_0_dict=head_0_stage_0_dict,
        head_0_stage_1_dict=head_0_stage_1_dict,
        head_1_stage_0_dict=head_1_stage_0_dict,
        head_1_stage_1_dict=head_1_stage_1_dict,
        head_0_stage_0_keys=head_0_stage_0_keys,
        head_0_stage_1_keys=head_0_stage_1_keys,
        head_1_stage_0_keys=head_1_stage_0_keys,
        head_1_stage_1_keys=head_1_stage_1_keys,
        head_0_stage_0_idxs=head_0_merge_idx_stage_1,
        head_0_stage_1_idxs=head_0_merge_idx_stage_2,
        head_1_stage_0_idxs=head_1_merge_idx_stage_1,
        head_1_stage_1_idxs=head_1_merge_idx_stage_2,
        head_0_merge=head_0_merge,
        head_1_merge=head_1_merge,
        head_merge=head_merge,
        loss_func=arch_info['loss_func'],
        num_classes=num_classes,
        loss_weight=loss_weight,
        ignore_value=cfg_seg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
    )


def build_stage_spatial_ops(stages, groups, spatial_ops, channels, input_img_size, stage_strides):
    """
    This method focus on build the feature group convolution. This used for RegNet like backbone which contains four
    different feature map stages and mainly used for segmentation which means s2, s3, s4 have the same stride.
    :param stages: The backbone output feature map used in this function
    :param groups: The split of feature into gropus
    :param spatial_ops: The random architecture sampled to build the final segmentation head.
    :param channels: Backbone model type used in this convolution
    :param input_img_size:
    :return:
    """
    begin_idx = get_begin_idx(stages)
    assert groups == len(spatial_ops), \
        "The number of architecture group has to be equal with spational operations"
    stage_ops_dict = {}
    stage_spatial_size = {f"s{idx+1}": (int(input_img_size[0] / stride), int(input_img_size[1] / stride))
                          for idx, stride in enumerate(stage_strides)}
    for s in stages:
        stage_stage_op_dict = {}
        ch_num = channels[begin_idx]//groups
        for j, op in enumerate(spatial_ops):
            layer_type, kvarge = build_op_parameters(op, ch_num, ch_num, stage_spatial_size[s])
            stage_stage_op_dict[f'g{j}'] = build_seg_nas_layers(layer_type, **kvarge)
        stage_ops_dict[s] = stage_stage_op_dict
        begin_idx += 1
    return stage_ops_dict


def build_channel_merge_ops(arch_op, channels, stages, input_img_size, stage_strides):
    begin_idx = get_begin_idx(stages)
    stage_channel_op_dict = {}
    stage_spatial_size = {f"s{idx+1}": (int(input_img_size[0] / stride), int(input_img_size[1] / stride))
                          for idx, stride in enumerate(stage_strides)}
    for s in stages:
        ch_num = channels[begin_idx]
        layer_type, kvarge = build_op_parameters(arch_op, ch_num, ch_num, stage_spatial_size[s])
        begin_idx += 1
        stage_channel_op_dict[f'{s}_channel'] = build_seg_nas_layers(layer_type, **kvarge)
    return stage_channel_op_dict


def build_stage_ops(stage_op_lists, channels, stages, input_img_size, stage_strides):
    begin_idx = get_begin_idx(stages)
    channels = channels[begin_idx:]

    stage_spatial_size = {f"s{idx+1}": (int(input_img_size[0] / stride), int(input_img_size[1] / stride))
                          for idx, stride in enumerate(stage_strides)}
    stage_op_dict = {}
    for idx, s in enumerate(stages):
        layer_type, kvarge = build_op_parameters(stage_op_lists[idx], channels[idx], channels[idx],
                                                 stage_spatial_size[s])
        stage_op_dict[f'{s}_feature_op'] = build_seg_nas_layers(layer_type, **kvarge)
    return stage_op_dict


def build_stage_merge_ops(head_0_merge_idx_stage_1, head_0_merge_idx_stage_2, head_1_merge_idx_stage_1,
                          head_1_merge_idx_stage_2, channels, stages, input_img_size, stage_strides, stage_merge_ratio):
    begin_idx = get_begin_idx(stages)
    channels = channels[begin_idx:]

    stage_spatial_size = {f"s{idx+1}": (int(input_img_size[0] / stride), int(input_img_size[1] / stride))
                          for idx, stride in enumerate(stage_strides)}

    head_0_stage_0_dict, head_0_stage_0_keys = build_feature_merge_stage_1(features_lists=head_0_merge_idx_stage_1,
                                                                           channels=channels,
                                                                           stage_merge_ratio=stage_merge_ratio,
                                                                           stage_spatial_size=stage_spatial_size,
                                                                           name="head_0_stage_0")

    head_0_stage_1_dict, head_0_stage_1_keys = build_feature_merge_stage_2(features_lists=head_0_merge_idx_stage_2,
                                                                           stage_0_info=head_0_stage_0_dict,
                                                                           stage_0_keys=head_0_stage_0_keys,
                                                                           stage_merge_ratio=stage_merge_ratio,
                                                                           name="head_0_stage_1")

    head_1_stage_0_dict, head_1_stage_0_keys = build_feature_merge_stage_1(features_lists=head_1_merge_idx_stage_1,
                                                                           channels=channels,
                                                                           stage_merge_ratio=stage_merge_ratio,
                                                                           stage_spatial_size=stage_spatial_size,
                                                                           name="head_1_stage_0")

    head_1_stage_1_dict, head_1_stage_1_keys = build_feature_merge_stage_2(features_lists=head_1_merge_idx_stage_2,
                                                                           stage_0_info=head_1_stage_0_dict,
                                                                           stage_0_keys=head_1_stage_0_keys,
                                                                           stage_merge_ratio=stage_merge_ratio,
                                                                           name="head_1_stage_1")

    return head_0_stage_0_dict, head_0_stage_1_dict, head_1_stage_0_dict, head_1_stage_1_dict, \
           head_0_stage_0_keys, head_0_stage_1_keys, head_1_stage_0_keys, head_1_stage_1_keys


def build_feature_merge_stage_1(features_lists, channels, stage_merge_ratio, stage_spatial_size, name):
    stage_merge_dict = {}
    keys = []
    for idx, feature_idxs in enumerate(features_lists):
        feature_1_channels = channels[feature_idxs[0]]
        feature_2_channels = channels[feature_idxs[1]]

        out_channels = (int(feature_1_channels * stage_merge_ratio), int(feature_2_channels * stage_merge_ratio))

        feature_1_spatial_size = stage_spatial_size[f"s{feature_idxs[0]+1}"]
        feature_2_spatial_size = stage_spatial_size[f"s{feature_idxs[1]+1}"]

        kvargs = {
            "in_channels": (feature_1_channels, feature_2_channels),
            "out_channels": out_channels,
            "spatial_sizes": (feature_1_spatial_size, feature_2_spatial_size),
            "name": name
        }

        stage_merge_dict[f"{name}_{idx}_feature_merge"] = build_seg_nas_layers("FeatureMerge", **kvargs)
        keys.append(f"{name}_{idx}_feature_merge")
    return stage_merge_dict, keys


def build_feature_merge_stage_2(features_lists, stage_0_info, stage_0_keys, stage_merge_ratio, name):
    stage_merge_dict = {}
    keys = []
    for idx, feature_idxs in enumerate(features_lists):
        feature_key_0 = stage_0_keys[feature_idxs[0]]
        feature_key_1 = stage_0_keys[feature_idxs[1]]

        feature_info_0 = stage_0_info[feature_key_0].get_sizes()
        feature_info_1 = stage_0_info[feature_key_1].get_sizes()

        feature_0_channels = feature_info_0["out_channels"]
        feature_1_channels = feature_info_1["out_channels"]

        out_channels = (int(feature_0_channels * stage_merge_ratio), int(feature_1_channels * stage_merge_ratio))

        feature_0_spatial_size = feature_info_0["out_spatial_size"]
        feature_1_spatial_size = feature_info_1["out_spatial_size"]

        kvargs = {
            "in_channels": (feature_0_channels, feature_1_channels),
            "out_channels": out_channels,
            "spatial_sizes": (feature_0_spatial_size, feature_1_spatial_size),
            "name": name
        }

        stage_merge_dict[f"{name}_{idx}_feature_merge"] = build_seg_nas_layers("FeatureMerge", **kvargs)
        keys.append(f"{name}_{idx}_feature_merge")
    return stage_merge_dict, keys


def build_head_merge_ops(head_merge_ops, head_0_info, head_1_info, seg_classes):
    if "ConcatHead" in head_merge_ops:
        in_channels = (head_0_info["out_channels"], head_1_info["out_channels"])
        spatial_size = max(head_0_info["out_spatial_size"][0], head_1_info["out_spatial_size"][0])
        kvargs = {
            "in_channel": in_channels,
            "spatial_size": (spatial_size, spatial_size),
            "seg_nums": seg_classes
        }
        head_merge_op = build_seg_nas_layers("ConcatHead", **kvargs)
    elif "GlobalSEHead" in head_merge_ops:
        layer_type, layer_ratio = head_merge_ops.split("_")
        layer_ratio = int(layer_ratio)
        in_channels = (head_0_info["out_channels"], head_1_info["out_channels"])
        spatial_size = max(head_0_info["out_spatial_size"][0], head_1_info["out_spatial_size"][0])
        kvargs = {
            "in_channel": in_channels,
            "spatial_size": (spatial_size, spatial_size),
            "seg_nums": seg_classes,
            "r": layer_ratio
        }
        head_merge_op = build_seg_nas_layers("GlobalSEHead", **kvargs)
    else:
        raise ValueError(f"Head merge type {head_merge_ops} does not support at present.")
    return head_merge_op


def build_op_parameters(layer_info, in_ch_num, out_ch_num, stage_spatial_size):
    if "Zero" in layer_info:
        return "Zero", {"ch_num": out_ch_num}
    elif "Conv2d" in layer_info:
        layer_type, kernel_size = layer_info.split("_")
        kernel_size = int(kernel_size)
        if kernel_size == 3:
            return layer_type, {"C_in": in_ch_num,
                                "C_out": out_ch_num,
                                "kernel_size": (kernel_size, kernel_size),
                                "stride": (1, 1),
                                "padding": (1, 1)}
        elif kernel_size == 1:
            return layer_type, {"C_in": in_ch_num,
                                "C_out": out_ch_num,
                                "kernel_size": (kernel_size, kernel_size),
                                "stride": (1, 1),
                                "padding": (0, 0)}
    elif "AdaptiveAvgPool" in layer_info:
        layer_type, ratio = layer_info.split("_")
        ratio = int(ratio)/100
        return layer_type, {"in_channel_size": (in_ch_num, out_ch_num),
                            "spatial_size": [int(s*ratio) for s in stage_spatial_size]
                            }
    elif "DilConv" in layer_info:
        layer_type, dilation_rate = layer_info.split("_")
        dilation_rate = int(dilation_rate)
        return layer_type, {"C_in": in_ch_num,
                            "C_out": out_ch_num,
                            "kernel_size": (3, 3),
                            "stride": (1, 1),
                            "padding": (dilation_rate, dilation_rate),
                            "dilation": (dilation_rate, dilation_rate)}
    elif "SEAttention" in layer_info and "SEAttentionStandard" not in layer_info:
        layer_type, ratio = layer_info.split("_")
        return layer_type, {"ch_num": in_ch_num,
                            "r": int(ratio)}
    elif "SEAttentionStandard" in layer_info:
        layer_type, ratio = layer_info.split("_")
        return layer_type, {"ch_num": in_ch_num,
                            "r": int(ratio)}
    elif "SelfAttention" in layer_info:
        return "SelfAttention",  {"in_channels": in_ch_num}


def get_begin_idx(stages):
    if 's1' in stages:
        begin_idx = 0
    elif 's2' in stages:
        begin_idx = 1
    elif 's3' in stages:
        begin_idx = 2
    elif "s4" in stages:
        begin_idx = 3
    else:
        raise ValueError('At least stage s3 should in the stage list!')
    return begin_idx


def get_img_hw_ratio(dataset_type):
    if 'cityscapes' in dataset_type:
        # h=1, w=2
        return [1, 2]
    else:
        return [1, 1]