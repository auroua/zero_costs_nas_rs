import os
import json
import numpy as np
import pylab as mpl
import seaborn as sns
import matplotlib.pyplot as plt


# sns.set_theme(style="ticks", palette="pastel")


mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def parse_json(json_file):
    with open(json_file, 'r') as f:
        all_lines = list(f.readlines())
        load_dict = json.loads(all_lines[-1])
        if "sem_seg/mIoU" not in load_dict:
            print(json_file)
            return None
        else:
            return [load_dict["sem_seg/mIoU"], load_dict["sem_seg/IoU-road"]]


def get_order(order_file_path):
    key_list_ordered = []
    with open(order_file_path, 'r') as f:
        for line in f:
            str_list = line.split(": ")
            key_str = str_list[2].split(",")[0]
            key_list_ordered.append(key_str)
    return key_list_ordered


def draw_perf_hist(perf_list):
    plt.hist(perf_list, bins=50, rwidth=0.8, range=(0, 100), align='right', density=True)
    # plt.ylim(0, 1)
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 101, 10))
    plt.xlabel("Accuracy [%]")
    plt.ylabel("Probability")
    # plt.legend()
    # data = {"perf": perf_list}
    # sns.histplot(data=data, x="perf", bins=20, stat="probability", kde=True)


def draw_perf_sequence(perf_list):
    seq_split = []
    for i in range(20):
        seq_split.append(perf_list[i*10:i*10+10])
    idx_list = []
    for i in range(20):
        if i == 2:
            idx_list.extend([(i+1)*10]*9)
        else:
            idx_list.extend([(i+1)*10]*10)
    min_val_list = []
    mean_list = []
    max_val_list = []
    for seq in seq_split:
        min_val_list.append(np.min(seq))
        mean_list.append(np.mean(seq))
        max_val_list.append(np.max(seq))

    data = {
        "budget": np.array(idx_list),
        "val_list": np.array(perf_list)
    }
    sns.boxplot(x="budget", y="val_list",
                data=data).set(xlabel='Number of Samples',
                               ylabel='Accuracy [%]')
    plt.show()


if __name__ == '__main__':
    searched_files = "E/npenas_seg101_mass_road_3"
    order_file_path = "D/WorkSpaces_Python/nas_seg_detectron2/tools_nas/performance_order.txt"
    searched_arch_metrics_json = [os.path.join(searched_files, f, "metrics.json") for f in os.listdir(searched_files) if
                             os.path.isdir(os.path.join(searched_files, f))]
    perf_dict = {}
    for metric_json in searched_arch_metrics_json:
        performance = parse_json(metric_json)
        if performance:
            perf_dict[metric_json.split("\\")[-2]] = performance

    key_order = get_order(order_file_path)
    miou_list = []
    iou_list = []
    for k in key_order:
        if k in perf_dict:
            miou_list.append(perf_dict[k][0])
            iou_list.append(perf_dict[k][1])

    plt.figure(figsize=(8, 5), dpi=600)
    draw_perf_hist(miou_list)
    # draw_perf_sequence(miou_list)

    plt.show()