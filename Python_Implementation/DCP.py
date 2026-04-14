import torch
from torch import nn


def get_conv_layer_weights(model, layer_name):
    try:
        layer = dict(model.named_modules())[layer_name]
        if isinstance(layer, (nn.Conv1d, nn.Linear)):
            return layer
        else:
            raise ValueError(f"The layer named '{layer_name}' is not supported.")
    except KeyError:
        raise ValueError(f"No layer named '{layer_name}' found in the model.")

def prunel1(model, layer_name, prune_rate):
    """
    Direct L1 pruning:
    prune the weights with the smallest absolute values in the target layer.

    Args:
        model: PyTorch model
        layer_name: target layer name
        prune_rate: ratio of weights to prune in this layer, in (0, 1]
    """
    if not (0 < prune_rate <= 1):
        raise ValueError("prune_rate must be in (0, 1].")

    layer = get_conv_layer_weights(model, layer_name)
    weights = layer.weight

    with torch.no_grad():
        flat_weights = weights.view(-1)
        total_num = flat_weights.numel()

        prune_num = max(1, int(total_num * prune_rate))
        prune_num = min(prune_num, total_num)

        # L1 criterion: smallest absolute values
        scores = torch.abs(flat_weights)
        prune_idx = torch.argsort(scores)[:prune_num]

        mask = torch.ones_like(flat_weights)
        mask[prune_idx] = 0

        flat_weights[prune_idx] = 0
        mask = mask.view_as(weights)

    def mask_hook(grad):
        return grad * mask

    layer.weight.register_hook(mask_hook)

def prunel2(model, layer_name, prune_rate):
    """
    Global L2 pruning on individual weights.
    Prune weights with the smallest squared magnitude (equivalent ranking to abs(weights)).
    """
    if not (0 < prune_rate <= 1):
        raise ValueError("prune_rate must be in (0, 1].")

    layer = get_conv_layer_weights(model, layer_name)
    weights = layer.weight

    with torch.no_grad():
        flat_weights = weights.view(-1)
        total_num = flat_weights.numel()

        prune_num = max(1, int(total_num * prune_rate))
        prune_num = min(prune_num, total_num)

        l2_scores = flat_weights.pow(2)
        prune_idx = torch.argsort(l2_scores)[:prune_num]

        mask = torch.ones_like(flat_weights)
        mask[prune_idx] = 0
        flat_weights[prune_idx] = 0
        mask = mask.view_as(weights)

    def mask_hook(grad):
        return grad * mask

    layer.weight.register_hook(mask_hook)

def prunerd(model, layer_name, prune_rate):
    """
    Random weight pruning:
    randomly prune a ratio of weights in the target layer.

    Args:
        model: PyTorch model
        layer_name: target layer name
        prune_rate: ratio of weights to prune in this layer, in (0, 1]
    """
    if not (0 < prune_rate <= 1):
        raise ValueError("prune_rate must be in (0, 1].")

    layer = get_conv_layer_weights(model, layer_name)
    weights = layer.weight

    with torch.no_grad():
        flat_weights = weights.view(-1)
        total_num = flat_weights.numel()

        prune_num = max(1, int(total_num * prune_rate))
        prune_num = min(prune_num, total_num)

        # random indices
        prune_idx = torch.randperm(total_num, device=flat_weights.device)[:prune_num]

        mask = torch.ones_like(flat_weights)
        mask[prune_idx] = 0

        flat_weights[prune_idx] = 0
        mask = mask.view_as(weights)

    def mask_hook(grad):
        return grad * mask

    layer.weight.register_hook(mask_hook)

def prune(model, layer_name, select_rate, prune_rate, mode=0):
    """
    两步剪枝（与你的描述一致）：
    1. 第一步：对权重本身排序（不是绝对值），选取较小的一部分
    2. 第二步：对第一步选出的权重求中值，按与中值的距离排序
       - 靠近中值的点保留
       - 远离中值的点剪掉

    参数:
        model: 模型
        layer_name: 层名
        select_rate: 第一步选取比例
        prune_rate: 第二步在第一步候选集合中的剪枝比例
        mode:
            0: 你的方法
            1: 在候选集合里直接随机剪枝
            2: 在候选集合里直接剪掉最小值
    """
    layer = get_conv_layer_weights(model, layer_name)
    weights = layer.weight

    with torch.no_grad():
        device = weights.device
        original_shape = weights.shape

        # 拉平成一维，按单个权重元素处理
        flat_weights = weights.view(-1)
        total_num = flat_weights.numel()

        # 第一步：选取“较小”的一部分（按权重值本身排序，不取绝对值）
        select_num = int(total_num * select_rate)
        if select_num <= 0:
            raise ValueError("select_rate is too small, selected number is 0.")

        candidate_indices = torch.argsort(flat_weights)[:select_num]
        candidate_values = flat_weights[candidate_indices]

        # 第二步：在候选集合中进一步决定剪谁
        prune_num = int(select_num * prune_rate)
        if prune_num <= 0:
            raise ValueError("prune_rate is too small, pruned number is 0.")
        prune_num = min(prune_num, select_num)

        if mode == 0:
            # 计算候选集合中值
            median_val = torch.median(candidate_values)

            # 与中值的距离
            distances = torch.abs(candidate_values - median_val)

            # 你的描述：靠近中值的保留 => 剪掉远离中值的
            # 所以这里取“距离最大”的 prune_num 个元素进行剪枝
            sorted_idx = torch.argsort(distances, descending=True)
            prune_indices = candidate_indices[sorted_idx[:prune_num]]

        elif mode == 1:
            # 在候选集合中随机剪枝
            rand_perm = torch.randperm(select_num, device=device)
            prune_indices = candidate_indices[rand_perm[:prune_num]]

        elif mode == 2:
            # 在候选集合中继续剪掉最小的 prune_num 个
            sorted_idx = torch.argsort(candidate_values)
            prune_indices = candidate_indices[sorted_idx[:prune_num]]

        else:
            raise ValueError("mode must be 0, 1, or 2.")

        # 构造 mask
        mask = torch.ones_like(flat_weights, device=device)
        mask[prune_indices] = 0

        # 执行剪枝
        flat_weights[prune_indices] = 0

        # 恢复形状
        mask = mask.view(original_shape)

    def mask_hook(grad):
        return grad * mask

    layer.weight.register_hook(mask_hook)