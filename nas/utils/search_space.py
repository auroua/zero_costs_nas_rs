import numpy as np
import torch
import math
from collections import defaultdict
from nas.utils.algos import quick_sort_list
import random
import os


def find_isolate_node(matrix):
    node_list = []
    for i in range(len(matrix)):
        if np.all(matrix[i, :] == 0) and np.all(matrix[:, i] == 0):
            if i == 0:
                continue
            matrix[0, i] = 1
            node_list.append(i)
    return node_list


def init_darts_folder(save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(os.path.join(save_dir, 'model_pkl')):
        os.mkdir(os.path.join(save_dir, 'model_pkl'))
    if not os.path.exists(os.path.join(save_dir, 'results')):
        os.mkdir(os.path.join(save_dir, 'results'))
    if not os.path.exists(os.path.join(save_dir, 'pre_train_models')):
        os.mkdir(os.path.join(save_dir, 'pre_train_models'))


def init_seg_folder(save_dir, nas_search_strategy):
    if not save_dir:
        save_dir = nas_search_strategy + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


def arch2graph(matrix, ops, num_vertices, node_vec_len, ops_list, is_idx=False):
    node_feature = torch.zeros(num_vertices, node_vec_len)
    if isinstance(matrix, torch.Tensor):
        edges = int(torch.sum(matrix).item())
    else:
        edges = int(np.sum(matrix))
    edge_idx = torch.zeros(2, edges)
    counter = 0
    for i in range(num_vertices):
        if is_idx:
            idx = int(ops[i].item())
        else:
            idx = ops_list.index(ops[i])
        node_feature[i, idx] = 1
        for j in range(num_vertices):
            if matrix[i, j] == 1:
                edge_idx[0, counter] = i
                edge_idx[1, counter] = j
                counter += 1
    return edge_idx, node_feature


def arch2edge_idx(matrix, num_vertices):
    if isinstance(matrix, torch.Tensor):
        edges = int(torch.sum(matrix).item())
    else:
        edges = int(np.sum(matrix))
    edge_idx = torch.zeros(2, edges)
    counter = 0
    for i in range(num_vertices):
        for j in range(num_vertices):
            if matrix[i, j] == 1:
                edge_idx[0, counter] = i
                edge_idx[1, counter] = j
                counter += 1
    return edge_idx


def arch2node_feature(ops, num_vertices, node_vec_len, ops_list, is_idx=False):
    node_feature = torch.zeros(num_vertices, node_vec_len)
    for i in range(num_vertices):
        if is_idx:
            idx = int(ops[i].item())
        else:
            idx = ops_list.index(ops[i])
        node_feature[i, idx] = 1
    return node_feature


def edit_distance_normalization(path_encoding_1, path_endocing_2, num_nodes):
    distance = np.sum(np.array(path_encoding_1) != np.array(path_endocing_2)) * 1.0
    distance = math.exp(-1.*(distance/num_nodes))
    return distance


def edit_distance(path_encoding_1, path_endocing_2):
    distance = np.sum(np.array(path_encoding_1) != np.array(path_endocing_2)) * 1.0
    return distance


def generate_min_vals(p1, p2, return_matrix=False):
    p1 = p1.view(1, p1.size(0), p1.size(1))
    p2 = p2.view(p2.size(0), 1, p2.size(1))
    dist = torch.sum(torch.abs(p1 - p2), dim=-1).int().T
    eigen_index = [i for i in range(dist.size(0))]
    dist[eigen_index, eigen_index] = 100
    min_vals, min_indices = torch.min(dist, dim=1)
    if return_matrix:
        return dist
    else:
        return min_vals, min_indices


def analysis_matrix(dist_matrix):
    dist_matrix_np = dist_matrix.numpy()
    for i in range(dist_matrix_np.shape[0]):
        min_val = np.min(dist_matrix_np[i, :])
        print(min_val, np.sum(dist_matrix_np[i, :] == min_val))


def get_paths_seq_aware(matrix, ops, num_vertices):
    paths = []
    paths_idx = []
    for j in range(0, num_vertices):
        paths.append([[]]) if matrix[0][j] else paths.append([])
        paths_idx.append([[]]) if matrix[0][j] else paths_idx.append([])
    for i in range(1, num_vertices - 1):
        for j in range(1, num_vertices):
            if matrix[i][j]:
                for ids, path in enumerate(paths[i]):
                    paths[j].append([*path, ops[i]])
                    paths_idx[j].append([*paths_idx[i][ids], i])
    return paths[-1], paths_idx[-1]


def sort_seqs_list(paths, paths_idx):
    seq_len_dict = defaultdict(list)
    for idx, p in enumerate(paths):
        seq_len_dict[len(p)].append(idx)
    k_sorted = sorted(list(seq_len_dict.keys()))
    sorted_idxs = []
    for k in k_sorted:
        paths_v = [(v_i, paths_idx[v_i]) for v_i in seq_len_dict[k]]
        sort_results = quick_sort_list(paths_v)
        sorted_idxs.extend([k[0] for k in sort_results])
    return [paths[idx] for idx in sorted_idxs], [paths_idx[idx] for idx in sorted_idxs]


def encode_path_seq_aware(matrix, ops, num_vertices, op_spots, ops_list, mapping, length):
    """ output one-hot encoding of paths """
    paths, paths_idx = get_paths_seq_aware(matrix=matrix, ops=ops,
                                           num_vertices=num_vertices)
    paths, paths_idx = sort_seqs_list(paths, paths_idx)
    vectors_list = []
    for (p_list, idx_list) in zip(paths, paths_idx):
        vec = np.zeros(op_spots*len(ops_list), dtype=np.int16)
        for p, ids in zip(p_list, idx_list):
            vec[(ids-1)*len(ops_list) + mapping[p]] = 1
        vectors_list.append(vec)
    path_encoding = np.array(vectors_list, dtype=np.int16)
    path_encoding = path_encoding.reshape((1, -1))[0]
    residual_len = length - path_encoding.shape[0]
    if residual_len != 0:
        residual_np = np.zeros(residual_len, dtype=np.int16)
        path_encoding = np.concatenate([path_encoding, residual_np])
    return path_encoding


def get_search_space_info(cfg):
    if cfg.SEARCH_SPACE.TYPE == 'NasBench101':
        num_vertices = cfg.SEARCH_SPACE.NASBENCH_101.NUM_VERTICES
        node_vec_len = cfg.SEARCH_SPACE.NASBENCH_101.GRAPH_NODE_DIM
        ops_list = cfg.NASBENCH_101.OPS
    elif cfg.SEARCH_SPACE.TYPE == 'NASBench201':
        num_vertices = cfg.SEARCH_SPACE.NASBENCH_201.NUM_VERTICES
        node_vec_len = cfg.SEARCH_SPACE.NASBENCH_201.GRAPH_NODE_DIM
        ops_list = cfg.NASBENCH_201.OPS
    else:
        raise NotImplemented(f'The search space {cfg.SEARCH_SPACE.SS} does not support at present!')
    return num_vertices, node_vec_len, ops_list


def get_input_dim(cfg):
    if cfg.SEARCH_SPACE.TYPE == 'NasBench101':
        input_dim = cfg.SEARCH_SPACE.NASBENCH_101.GRAPH_NODE_DIM
    elif cfg.SEARCH_SPACE.TYPE == 'NASBench201':
        input_dim = cfg.SEARCH_SPACE.NASBENCH_201.GRAPH_NODE_DIM
    elif cfg.SEARCH_SPACE.TYPE == "SEG101":
        input_dim = cfg.SEARCH_SPACE.SEG_101.GRAPH_NODE_DIM
    else:
        raise NotImplemented(f'The search space {cfg.SEARCH_SPACE.TYPE} is not supported at present!')
    middle_dim = cfg.PREDICTOR.DIM2
    num_classes = cfg.PREDICTOR.NUM_CLASSES
    return input_dim, middle_dim, num_classes


def dataset_split_idx(all_data, budget=None):
    idxs = list(range(len(all_data)))
    total_keys = list(all_data.keys())
    random.shuffle(idxs)
    train_data = [all_data[total_keys[k]] for k in idxs[:budget]]
    test_data = [all_data[total_keys[kt]] for kt in idxs[budget:]]
    return train_data, test_data