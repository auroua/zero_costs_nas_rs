import copy
import numpy as np
import sys


def _compare_lists(list1, list2):
    for (l1, l2) in zip(list1[1], list2[1]):
        if l1 == l2:
            continue
        elif l1 < l2:
            return True
        else:
            return False


# There are different ways to do a Quick Sort partition, this implements the
# Hoare partition scheme. Tony Hoare also created the Quick Sort algorithm.
def _partition_list(nums, low, high):
    # We select the middle element to be the pivot. Some implementations select
    # the first element or the last element. Sometimes the median value becomes
    # the pivot, or a random one. There are many more strategies that can be
    # chosen or created.
    pivot = nums[(low + high) // 2]
    i = low - 1
    j = high + 1
    while True:
        i += 1
        # while nums[i] < pivot:
        while _compare_lists(nums[i], pivot):
            i += 1

        j -= 1
        while _compare_lists(pivot, nums[j]):
            j -= 1

        if i >= j:
            return j

        # If an element at i (on the left of the pivot) is larger than the
        # element at j (on right right of the pivot), then swap them
        nums[i], nums[j] = nums[j], nums[i]


def quick_sort_list(nums):
    # Create a helper function that will be called recursively
    datas = copy.deepcopy(nums)

    def _quick_sort(items, low, high):
        if low < high:
            # This is the index after the pivot, where our lists are split
            split_index = _partition_list(items, low, high)
            _quick_sort(items, low, split_index)
            _quick_sort(items, split_index + 1, high)

    _quick_sort(datas, 0, len(datas) - 1)
    return datas


def acq_fn(predictions, explore_type='its', reverse=False):
    if explore_type != 'its_vae' and explore_type != 'its_vae_ensemble':
        predictions = np.array(predictions)
    # Thompson sampling (TS) acquisition function
    if explore_type == 'ts':
        rand_ind = np.random.randint(predictions.shape[0])
        ts = predictions[rand_ind,:]
        sorted_indices = np.argsort(ts)
    # Independent Thompson sampling (ITS) acquisition function
    elif explore_type == 'its':
        mean = np.mean(predictions, axis=0)
        std = np.sqrt(np.var(predictions, axis=0))
        samples = np.random.normal(mean, std)
        sorted_indices = np.argsort(samples)
    elif explore_type == 'its_vae':
        np_pred = predictions
        sorted_indices = np.argsort(np_pred)
    else:
        print('Invalid exploration type in meta neuralnet search', explore_type)
        sys.exit()

    return sorted_indices


if __name__ == '__main__':
    # Verify it works
    # random_list_of_nums = [22, 5, 1, 18, 99]
    random_list_of_nums = [(3, [1, 3, 5]), (2, [1, 3, 5]), (0, [2, 3, 4]),
                           (1, [3, 4, 6]), (4, [1, 2, 6]), (5, [1, 1, 1]), (6, [3, 2, 1])]
    results = quick_sort_list(random_list_of_nums)
    print(random_list_of_nums)
    print(results)