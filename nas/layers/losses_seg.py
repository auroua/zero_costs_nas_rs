import torch
import torch.nn.functional as F


def dice_loss(inputs, targets, return_type='value'):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        return_type: return a dict form or value form
    """
    batch_size, cat_num = inputs.size(0), inputs.size(1)

    # Avoid using background
    # Multiple class dice loss, reference from: https://www.jeremyjordan.me/semantic-segmentation/
    inputs = F.softmax(inputs, dim=1)[:, 1:, :, :]
    inputs = inputs.flatten(2)
    targets = target_onehot(targets, cat_num)[:, 1:, :, :]
    targets = targets.flatten(2)

    numerator = 2 * (inputs * targets).sum(2)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if return_type == 'dict':
        return {
            'loss': loss.sum() / (batch_size * (cat_num - 1))
        }
    else:
        return loss.sum() / (batch_size * (cat_num - 1))


def target_onehot(target_tensor, cat_num):
    # target_cat_tensor = torch.cat([(target_tensor == i).float() for i in range(cat_num)], dim=1)[:, 1:, :, :]
    if len(target_tensor.size()) == 3:
        target_tensor = target_tensor.unsqueeze(dim=1)
    b, c, h, w = target_tensor.size()
    target_one_hot = torch.zeros(size=(b, cat_num, h, w), device=target_tensor.device, dtype=torch.float32)
    target_one_hot.scatter_(dim=1, index=target_tensor, value=1)
    return target_one_hot