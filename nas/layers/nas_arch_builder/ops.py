# import torch
# from nas.layers.nas_arch_builder.nn_layers import DilConv, Zero, AdaptiveAvgPool, SepConv, \
#     ConcatLayer, FPNHead, ConcatHeadLayer
# from nas.layers.nas_arch_builder.nn_layers_attention import SelfAttention, SEAttention, SelfAttentionChannel, \
#     SelfGlobalAttentionChannel, GlobalSE, GlobalSEHead, SynthesizerRandomAttention, \
#     SynthesizerRandomAttentionMergeFeatureMap, SynthesizerRandomSelfAttention
#
#
# def build_none(in_ch_num, out_ch_num, spatial_size):
#     return Zero(out_ch_num)
#
#
# def build_avg_25(in_ch_num, out_ch_num, spatial_size):
#     output_spatial_size = [2, 2]
#     return AdaptiveAvgPool((in_ch_num, out_ch_num), output_spatial_size, spatial_size)
#
#
# def build_avg_5(in_ch_num, out_ch_num, spatial_size):
#     output_spatial_size = [5, 5]
#     return AdaptiveAvgPool((in_ch_num, out_ch_num), output_spatial_size, spatial_size)
#
#
# def build_avg_75(in_ch_num, out_ch_num, spatial_size):
#     output_spatial_size = [10, 10]
#     return AdaptiveAvgPool((in_ch_num, out_ch_num), output_spatial_size, spatial_size)
#
#
# def build_dila_conv_3(in_ch_num, out_ch_num, spatial_size):
#     return DilConv(in_ch_num, out_ch_num, kernel_size=3, stride=1, padding=0, dilation=3)
#
#
# def build_dila_conv_5(in_ch_num, out_ch_num, spatial_size):
#     return DilConv(in_ch_num, out_ch_num, kernel_size=3, stride=1, padding=0, dilation=5)
#
#
# def build_dila_conv_7(in_ch_num, out_ch_num, spatial_size):
#     return DilConv(in_ch_num, out_ch_num, kernel_size=3, stride=1, padding=0, dilation=7)
#
#
# def build_conv1x1(in_ch_num, out_ch_num, spatial_size):
#     return SepConv(in_ch_num, out_ch_num, kernel_size=1, stride=1, padding=0)
#
#
# def build_self_attention(in_ch_num, out_ch_num, spatial_size):
#     return SelfAttention(in_ch_num)
#
#
# def build_se(in_ch_num, out_ch_num, spatial_size):
#     return SEAttention(in_ch_num)
#
#
# def build_self_channel_attention(in_ch_num, out_ch_num, spatial_size):
#     return SelfAttentionChannel(in_ch_num)
#
#
# def build_synthesizer_random_self_attention(in_ch_num, out_ch_num, spatial_size):
#     return SynthesizerRandomSelfAttention(in_ch_num, out_ch_num, spatial_size)
#
#
# def build_synthesizer_random_attention(in_ch_num, out_ch_num, spatial_size):
#     return SynthesizerRandomAttention(in_ch_num, out_ch_num, spatial_size)
#
#
# def build_self_global_channel_attention(in_ch_num, out_ch_num, spatial_size):
#     return SelfGlobalAttentionChannel(in_ch_num)
#
#
# def build_global_se8_conv(in_ch_num, out_ch_num, spatial_size):
#     return GlobalSE(in_ch_num, out_ch_num, r=8)
#
#
# def build_synthesizer_random_attention_merge_head(in_ch_num, out_ch_num, spatial_size):
#     return SynthesizerRandomAttentionMergeFeatureMap(in_ch_num, out_ch_num, spatial_size)
#
#
# def build_global_se6_conv(in_ch_num, out_ch_num, spatial_size):
#     return GlobalSE(in_ch_num, out_ch_num, r=6)
#
#
# def build_global_se4_conv(in_ch_num, out_ch_num, spatial_size):
#     return GlobalSE(in_ch_num, out_ch_num, r=4)
#
#
# def build_fpn(in_ch_num, out_ch_num, spatial_size):
#     return FPNHead(in_ch_num, out_ch_num)
#
#
# def build_concat_conv(in_ch_num, out_ch_num, spatial_size):
#     return ConcatLayer(in_ch_num, out_ch_num)
#
#
# def build_concat_conv_head(in_ch_num, head_counts, seg_classes, input_size):
#     return ConcatHeadLayer(in_ch_num, head_counts, seg_classes, input_size)
#
#
# def build_global_se8_conv_head(in_ch_num, head_counts, seg_classes, input_size):
#     return GlobalSEHead(in_ch_num, head_counts, seg_classes, input_size, r=8)
#
#
# def build_context_seg_ops(op, in_ch_num, out_ch_num, spatial_size):
#     return eval(f"build_{op}")(in_ch_num, out_ch_num, spatial_size)
#
#
# def build_context_seg_head_merge_ops(op, in_ch_num, head_counts, seg_classes, input_size):
#     return eval(f"build_{op}")(in_ch_num, head_counts, seg_classes, input_size)
#
#
# if __name__ == '__main__':
#     op = 'dila_conv_3'
#     # op = 'self_channel_attention'
#     # op = 'self_attention'
#     in_channel = 32
#     ch_num = 96
#     spatial_size = (64, 128)
#     s2 = torch.randn(1, in_channel, spatial_size[0], spatial_size[1])
#     s3 = torch.randn(1, in_channel, spatial_size[0], spatial_size[1])
#     s4 = torch.randn(1, in_channel, spatial_size[0], spatial_size[1])
#
#     # layer = build_context_seg_ops(op, ch_num*3, 32, spatial_size)
#     # layer = build_context_seg_head_merge_ops(op, ch_num, 3, 20, (224, 512))
#     layer = build_context_seg_ops(op, in_channel, 20, spatial_size)
#
#     # print(layer((s2, s3, s4)).size())
#     # print(layer(s2))
#     print(layer(s2).size())
#
