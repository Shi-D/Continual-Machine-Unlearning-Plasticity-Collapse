"""
Continual Backprop: 神经元重置算法

核心原理：
1. 在传统梯度下降基础上，增加随机初始化的非梯度组件
2. 在每一步训练中，识别并重新初始化利用率低 (low-utility) 的神经元
3. 只有被定义为"成熟"且贡献效用最小的单元才会被重置
4. 输出权重设为零，防止干扰现有功能
5. 防止单元进入"休眠"状态，维持较高的表示多样性（高有效秩）

Reference: Continual Backprop (Dohare et al.)
"""
import torch
import torch.nn as nn
import numpy as np


class ContinualBackpropEngine:
    """
    Continual Backprop 引擎

    负责跟踪神经元的成熟度和效用，并在适当时机重置低效用神经元
    """

    def __init__(
        self,
        model,
        reset_ratio=0.001,           # 每步重置的神经元比例
        maturity_threshold=1000,     # 成熟度阈值（经过多少次更新后才算成熟）
        utility_decay=0.99,          # 效用值的指数衰减率
        device='cuda'
    ):
        """
        Args:
            model: 神经网络模型
            reset_ratio: 每步重置的神经元比例 (0-1)
            maturity_threshold: 成熟度阈值，神经元经过这么多次更新后才被视为"成熟"
            utility_decay: 效用值的指数移动平均衰减率
            device: 计算设备
        """
        self.model = model
        self.reset_ratio = reset_ratio
        self.maturity_threshold = maturity_threshold
        self.utility_decay = utility_decay
        self.device = device

        # 存储每个神经元的元信息
        self.neuron_info = {}  # {layer_name: {'utility': tensor, 'maturity': tensor}}

        # 初始化神经元跟踪
        self._init_neuron_tracking()

    def _init_neuron_tracking(self):
        """
        初始化神经元跟踪信息

        为每个包含神经元的层创建：
        - utility: 效用值（衡量神经元贡献）
        - maturity: 成熟度计数器
        """
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # 获取神经元数量（输出通道数）
                if isinstance(module, nn.Linear):
                    num_neurons = module.out_features
                else:  # Conv2d
                    num_neurons = module.out_channels

                self.neuron_info[name] = {
                    'utility': torch.ones(num_neurons, device=self.device),
                    'maturity': torch.zeros(num_neurons, device=self.device, dtype=torch.long),
                    'module': module
                }

    def compute_utility(self, module, activation):
        """
        计算神经元的效用值

        效用定义为：神经元激活值的绝对值 * 输出权重的范数
        这衡量了神经元对下一层输出的贡献程度

        Args:
            module: 当前层模块
            activation: 该层的激活输出 [batch, neurons, ...]

        Returns:
            utility: 每个神经元的效用值 [neurons]
        """
        if isinstance(module, nn.Linear):
            # activation: [batch, out_features]
            # 计算激活值的平均绝对值
            act_contribution = activation.abs().mean(dim=0)  # [out_features]

            # 获取输出权重范数（如果有下一层的话）
            weight_norm = module.weight.data.abs().mean(dim=1)  # [out_features]

            utility = act_contribution * weight_norm

        elif isinstance(module, nn.Conv2d):
            # activation: [batch, out_channels, H, W]
            # 计算每个通道的平均激活
            act_contribution = activation.abs().mean(dim=(0, 2, 3))  # [out_channels]

            # 权重范数
            weight_norm = module.weight.data.abs().mean(dim=(1, 2, 3))  # [out_channels]

            utility = act_contribution * weight_norm
        else:
            return None

        return utility.detach()

    def update_utility(self, layer_name, current_utility):
        """
        使用指数移动平均更新神经元效用值

        Args:
            layer_name: 层名称
            current_utility: 当前计算的效用值
        """
        if layer_name not in self.neuron_info:
            return

        info = self.neuron_info[layer_name]
        # 指数移动平均更新
        info['utility'] = (self.utility_decay * info['utility'] +
                          (1 - self.utility_decay) * current_utility)
        # 增加成熟度计数
        info['maturity'] += 1

    def get_neurons_to_reset(self, layer_name):
        """
        获取需要重置的神经元索引

        选择条件：
        1. 神经元已经成熟（maturity >= maturity_threshold）
        2. 效用值最低的那些神经元

        Args:
            layer_name: 层名称

        Returns:
            indices: 需要重置的神经元索引
        """
        if layer_name not in self.neuron_info:
            return []

        info = self.neuron_info[layer_name]
        utility = info['utility']
        maturity = info['maturity']

        # 找出成熟的神经元
        mature_mask = maturity >= self.maturity_threshold

        if not mature_mask.any():
            return []

        # 在成熟神经元中找效用最低的
        mature_utility = utility.clone()
        mature_utility[~mature_mask] = float('inf')  # 排除不成熟的

        # 计算要重置的数量
        num_mature = mature_mask.sum().item()
        num_to_reset = max(1, int(num_mature * self.reset_ratio))

        # 选择效用最低的神经元
        _, indices = torch.topk(mature_utility, num_to_reset, largest=False)

        return indices.tolist()

    def reset_neuron(self, layer_name, neuron_idx):
        """
        重置指定神经元

        操作：
        1. 随机重新初始化输入权重
        2. 将输出权重设为零（防止干扰现有功能）
        3. 重置成熟度计数器
        4. 重置效用值

        Args:
            layer_name: 层名称
            neuron_idx: 神经元索引
        """
        if layer_name not in self.neuron_info:
            return

        info = self.neuron_info[layer_name]
        module = info['module']

        with torch.no_grad():
            if isinstance(module, nn.Linear):
                # 重新初始化输入权重（该神经元对应的行）
                fan_in = module.weight.data.size(1)
                std = 1.0 / np.sqrt(fan_in)
                module.weight.data[neuron_idx].normal_(0, std)

                # 如果有 bias，重置对应的 bias
                if module.bias is not None:
                    module.bias.data[neuron_idx] = 0.0

            elif isinstance(module, nn.Conv2d):
                # 重新初始化输入权重（该通道对应的卷积核）
                fan_in = module.weight.data.size(1) * module.weight.data.size(2) * module.weight.data.size(3)
                std = 1.0 / np.sqrt(fan_in)
                module.weight.data[neuron_idx].normal_(0, std)

                if module.bias is not None:
                    module.bias.data[neuron_idx] = 0.0

        # 重置元信息
        info['maturity'][neuron_idx] = 0
        info['utility'][neuron_idx] = 1.0  # 重置为初始效用

    def reset_output_weights(self, layer_name, neuron_idx, next_module):
        """
        将神经元的输出权重设为零

        这确保重置的神经元不会立即影响网络输出，
        需要重新"学习"其贡献

        Args:
            layer_name: 当前层名称
            neuron_idx: 神经元索引
            next_module: 下一层模块
        """
        if next_module is None:
            return

        with torch.no_grad():
            if isinstance(next_module, nn.Linear):
                # 将下一层中对应该神经元的列设为零
                next_module.weight.data[:, neuron_idx] = 0.0

            elif isinstance(next_module, nn.Conv2d):
                # 将下一层中对应该通道的输入设为零
                next_module.weight.data[:, neuron_idx, :, :] = 0.0

    def step(self, activations_dict=None):
        """
        执行一步神经元重置

        在每个训练步骤后调用，用于：
        1. 更新神经元效用（如果提供了激活值）
        2. 识别并重置低效用神经元

        Args:
            activations_dict: {layer_name: activation_tensor} 每层的激活值（可选）

        Returns:
            reset_stats: 重置统计信息
        """
        reset_stats = {'total_reset': 0, 'layers': {}}

        # 获取所有层的顺序，用于找下一层
        layer_names = list(self.neuron_info.keys())

        for i, layer_name in enumerate(layer_names):
            info = self.neuron_info[layer_name]
            module = info['module']

            # 如果提供了激活值，更新效用
            if activations_dict and layer_name in activations_dict:
                current_utility = self.compute_utility(module, activations_dict[layer_name])
                if current_utility is not None:
                    self.update_utility(layer_name, current_utility)

            # 获取需要重置的神经元
            neurons_to_reset = self.get_neurons_to_reset(layer_name)

            if neurons_to_reset:
                # 找到下一层模块（用于重置输出权重）
                next_module = None
                if i + 1 < len(layer_names):
                    next_module = self.neuron_info[layer_names[i + 1]]['module']

                for neuron_idx in neurons_to_reset:
                    self.reset_neuron(layer_name, neuron_idx)
                    self.reset_output_weights(layer_name, neuron_idx, next_module)

                reset_stats['layers'][layer_name] = len(neurons_to_reset)
                reset_stats['total_reset'] += len(neurons_to_reset)

        return reset_stats

    def compute_effective_rank(self, activation):
        """
        计算激活的有效秩（衡量表示多样性）

        有效秩 = exp(H(σ))，其中 H 是归一化奇异值的熵

        Args:
            activation: 激活张量 [batch, features] 或 [batch, channels, H, W]

        Returns:
            effective_rank: 有效秩值
        """
        # 展平为 2D
        if activation.dim() > 2:
            activation = activation.flatten(start_dim=2).mean(dim=2)  # [batch, channels]

        # 计算 SVD
        try:
            U, S, V = torch.svd(activation)
        except:
            return 0.0

        # 归一化奇异值
        S_norm = S / S.sum()
        S_norm = S_norm[S_norm > 1e-10]  # 移除太小的值

        if len(S_norm) == 0:
            return 0.0

        # 计算熵
        entropy = -(S_norm * torch.log(S_norm)).sum()

        # 有效秩
        effective_rank = torch.exp(entropy).item()

        return effective_rank


class ActivationHook:
    """
    激活值捕获钩子

    用于捕获前向传播中的激活值，供 ContinualBackpropEngine 使用
    """

    def __init__(self, model, layer_names):
        """
        Args:
            model: 神经网络模型
            layer_names: 要捕获的层名称列表
        """
        self.model = model
        self.layer_names = layer_names
        self.activations = {}
        self.hooks = []

        self._register_hooks()

    def _register_hooks(self):
        """注册前向钩子"""
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                hook = module.register_forward_hook(self._create_hook(name))
                self.hooks.append(hook)

    def _create_hook(self, name):
        """创建钩子函数"""
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

    def clear(self):
        """清空激活值缓存"""
        self.activations = {}

    def remove(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def create_continual_backprop_engine(model, args):
    """
    工厂函数：创建 ContinualBackpropEngine 实例

    Args:
        model: 神经网络模型
        args: 包含配置参数的对象

    Returns:
        engine: ContinualBackpropEngine 实例
    """
    reset_ratio = getattr(args, 'cb_reset_ratio', 0.001)
    maturity_threshold = getattr(args, 'cb_maturity_threshold', 1000)
    utility_decay = getattr(args, 'cb_utility_decay', 0.99)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return ContinualBackpropEngine(
        model=model,
        reset_ratio=reset_ratio,
        maturity_threshold=maturity_threshold,
        utility_decay=utility_decay,
        device=device
    )
