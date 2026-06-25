import numpy as np
from torch.utils.data import DataLoader, Subset, dataset, Dataset, random_split
from tqdm import tqdm
import time
import torch
from collections import OrderedDict
import torch.nn as nn
import os
from copy import deepcopy
import torch.nn.functional as F
import utils

from methods.SFRon import SFRon
from methods.MUNBa import Munba
from methods.ContinualBackprop import ContinualBackpropEngine, ActivationHook, create_continual_backprop_engine




def adjust_learning_rate(epoch, lr_decay_epochs, lr_decay_rate,
                         sgda_learning_rate, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    new_lr = sgda_learning_rate
    if steps > 0:
        new_lr = sgda_learning_rate * (lr_decay_rate**steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    return new_lr


'''basic func'''
def training_step(model, batch, criterion):
    device = next(model.parameters()).device
    images, clabels = batch
    images, clabels = images.to(device), clabels.to(device)
    out = model(images)  # Generate predictions
    loss = criterion(out, clabels)  # Calculate loss

    return loss


def training_step_split(model, batch, criterion, beta, unlearn_method):
    device = next(model.parameters()).device
    images, clabels, labels = batch
    images, clabels, labels = images.to(device), clabels.to(device), labels.to(device)

    out = model(images)

    retain_mask = (labels == 0)
    forget_mask = (labels == 1)

    retain_logits = out[retain_mask]
    retain_clabels = clabels[retain_mask]
    loss_retain = criterion(retain_logits, retain_clabels)

    forget_logits = out[forget_mask]
    forget_clabels = clabels[forget_mask]
    loss_forget = criterion(forget_logits, forget_clabels)


    if unlearn_method in ['salun', 'randomlabel']:
        return loss_retain*(sum(retain_mask)/len(labels)), loss_forget*(sum(forget_mask)/len(labels))
    elif unlearn_method in ['ga_plus']:
        return loss_retain, -beta*loss_forget


def fit_one_cycle(
    train_loader, model, optimizer, criterion, scheduler, unlearn_method, mask, beta=0.15, args={}
):
    model.train()

    pbar = tqdm(train_loader, total=len(train_loader))
    for batch_i, batch in enumerate(pbar):


        if unlearn_method in ['ga_plus', 'salun', 'randomlabel']:
            loss_retain, loss_forget = training_step_split(model, batch, criterion, beta, unlearn_method)
            loss = loss_retain + loss_forget
        else:
            loss = training_step(model, batch, criterion)
            loss = -loss if unlearn_method == 'ga' else loss

        optimizer.zero_grad()
        loss.backward()

        if mask and unlearn_method == 'salun':
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]
        optimizer.step()

    scheduler.step()

    return beta


def baseline(
    loader,
    model,
    optimizer,
    criterion,
    lr_schedule,
    mask,
    args,
    **kwargs
):

    res = fit_one_cycle(
        loader,
        model,
        optimizer,
        criterion,
        lr_schedule,
        args.unlearn_method,
        mask,
        args.beta,
        args
    )
    return res

def retrain(
    loader,
    model,
    optimizer,
    criterion,
    lr_schedule,
    mask,
    args,
    **kwargs
):

    res = fit_one_cycle(
        loader,
        model,
        optimizer,
        criterion,
        lr_schedule,
        args.unlearn_method,
        mask,
        args.beta,
        args
    )
    return res


def finetune(
    loader,
    model,
    optimizer,
    criterion,
    lr_schedule,
    mask,
    args,
    **kwargs
):
    res = fit_one_cycle(
        loader,
        model,
        optimizer,
        criterion,
        lr_schedule,
        args.unlearn_method,
        mask,
        args.beta,
        args
    )
    return res

def finetune2(
    loader,
    model,
    optimizer,
    criterion,
    lr_schedule,
    mask,
    args,
    **kwargs
):
    res = fit_one_cycle(
        loader,
        model,
        optimizer,
        criterion,
        lr_schedule,
        args.unlearn_method,
        mask,
        args.beta,
        args
    )
    return res

def ga(
    loader,
    model,
    optimizer,
    criterion,
    lr_schedule,
    mask,
    args,
    **kwargs
):

    res = fit_one_cycle(
        loader,
        model,
        optimizer,
        criterion,
        lr_schedule,
        args.unlearn_method,
        mask,
        args.beta,
        args
    )
    return res

def ga_plus(
    loader,
    model,
    optimizer,
    criterion,
    lr_schedule,
    mask,
    args,
    **kwargs
):
    res = fit_one_cycle(
        loader,
        model,
        optimizer,
        criterion,
        lr_schedule,
        args.unlearn_method,
        mask,
        args.beta,
        args
    )
    return res



def randomlabel(
    loader,
    model,
    optimizer,
    criterion,
    lr_schedule,
    mask,
    args,
    num_classes=10,
    **kwargs
):
    res = fit_one_cycle(
        loader,
        model,
        optimizer,
        criterion,
        lr_schedule,
        args.unlearn_method,
        mask,
        args.beta,
        args
    )
    return res

def salun(
    loader,
    model,
    optimizer,
    criterion,
    lr_schedule,
    mask,
    args,
    num_classes=10,
    **kwargs
):
    res = fit_one_cycle(
        loader,
        model,
        optimizer,
        criterion,
        lr_schedule,
        args.unlearn_method,
        mask,
        args.beta,
        args
    )
    return res



def sfron(
    loaders,
    model,
    optimizer,
    criterion,
    lr_schedule,
    mask,
    args,
    num_classes=10,
    **kwargs
):
    device = next(model.parameters()).device
    save_dir = './sfron_save_dir/'
    print('loaders', loaders)
    if args.unlearn_type == 'random':
        save_dir = os.path.join(args.model +'_' + args.dataset,
                            args.unlearn_method +'_new_forget' + str(int(100-args.retain_ratio*100)) +'_epoch' + str(args.epochs))

        train_retain_dl, train_forget_dl, test_retain_dl, test_forget_dl = (
            loaders['retain_dl'], loaders['forget_dl'], loaders['test_dl'], None)
    elif args.unlearn_type == 'class':
        save_dir = os.path.join(args.model + '_' + args.dataset,
                                args.unlearn_method + '_new_forget_class_epoch' + str(args.epochs))
        train_retain_dl, train_forget_dl, test_retain_dl, test_forget_dl = (
            loaders['retain_dl'], loaders['forget_dl'], loaders['test_dl'], None)

    print('save_dir', save_dir)
    os.makedirs(save_dir, exist_ok=True)

    loss_function = nn.CrossEntropyLoss()
    unlearn_dataloaders = OrderedDict(
        forget_train=train_forget_dl,
        retain_train=train_retain_dl,
        forget_valid=test_forget_dl,
        retain_valid=test_retain_dl
    )
    print('sfronsfron')
    unlearn_method = SFRon(model, loss_function, save_dir, args)
    print('sfronsfron prepare_unlearn')
    unlearn_method.prepare_unlearn(unlearn_dataloaders)
    model = unlearn_method.get_unlearned_model()


    return model


def munba(
    loaders,
    model,
    optimizer,
    criterion,
    lr_schedule,
    mask,
    args,
    num_classes=10,
    **kwargs
):
    train_retain_dl, train_forget_dl, test_retain_dl = loaders['retain_dl'], loaders['forget_dl'], loaders['test_dl']
    device = 'cuda'
    unlearn_data_loaders = OrderedDict(
        retain=train_retain_dl, forget=train_forget_dl, val=test_retain_dl, test=test_retain_dl
    )

    criterion = nn.CrossEntropyLoss()
    Munba(unlearn_data_loaders, model, criterion, num_classes, args)


    return model



def distill_kl_loss(y_s, y_t, T):
    """Distilling the Knowledge in a Neural Network"""
    p_s = F.log_softmax(y_s / T, dim=1)
    p_t = F.softmax(y_t / T, dim=1)
    loss = F.kl_div(p_s, p_t, reduction='batchmean') * (T**2)
    return loss



class LossData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if index < self.forget_len:
            x, y = self.forget_data[index]
            label = 1
            return x, y, label
        else:
            x, y = self.retain_data[index - self.forget_len]
            label = 0
            return x, y, label

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item(), len(preds), torch.tensor(torch.sum(preds == labels).item() / len(labels)) * 100

def evaluate_acc_batch(model, batch, device):
    images, clabels = batch
    images, clabels = images.to(device), clabels.to(device)
    out = model(images)
    return accuracy(out, clabels)

def evaluate_acc(model, val_loader, device):
    model.eval()
    corr, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            corr_, total_, _ = evaluate_acc_batch(model, batch, device)
            corr += corr_
            total += total_
    torch.cuda.empty_cache()
    return corr/total

def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


# ============================================================
# Continual Backprop: 神经元重置算法
# ============================================================

def continual_backprop(
    loader,
    model,
    optimizer,
    criterion,
    lr_schedule,
    mask,
    args,
    cb_engine=None,
    **kwargs
):
    """
    Continual Backprop 训练函数

    在标准梯度下降基础上，增加神经元重置机制：
    1. 识别利用率低的神经元
    2. 重置成熟且低效用的神经元
    3. 将输出权重设为零防止干扰
    4. 维持网络可塑性

    Args:
        loader: 数据加载器
        model: 神经网络模型
        optimizer: 优化器
        criterion: 损失函数
        lr_schedule: 学习率调度器
        mask: 梯度掩码（可选）
        args: 配置参数
        cb_engine: ContinualBackpropEngine 实例（可选，如果不提供则创建新的）

    Returns:
        cb_engine: 更新后的 ContinualBackpropEngine 实例
    """
    device = next(model.parameters()).device
    model.train()

    # 创建或使用已有的 Continual Backprop 引擎
    if cb_engine is None:
        cb_engine = create_continual_backprop_engine(model, args)

    # 设置激活值捕获钩子
    layer_names = list(cb_engine.neuron_info.keys())
    activation_hook = ActivationHook(model, layer_names)

    # 获取配置参数
    beta = getattr(args, 'beta', 0.15)
    unlearn_method = getattr(args, 'unlearn_method', 'continual_backprop')
    reset_interval = getattr(args, 'cb_reset_interval', 10)  # 每多少个 batch 执行一次重置

    total_reset = 0
    effective_ranks = []

    pbar = tqdm(loader, total=len(loader), desc='Continual Backprop')
    for batch_i, batch in enumerate(pbar):
        # 清空之前的激活值
        activation_hook.clear()

        # 前向传播
        if len(batch) == 3:  # (images, clabels, labels) 格式
            images, clabels, labels = batch
            images, clabels, labels = images.to(device), clabels.to(device), labels.to(device)

            out = model(images)

            retain_mask = (labels == 0)
            forget_mask = (labels == 1)

            # 计算 retain 和 forget loss
            retain_logits = out[retain_mask]
            retain_clabels = clabels[retain_mask]
            loss_retain = criterion(retain_logits, retain_clabels) if retain_mask.any() else 0

            forget_logits = out[forget_mask]
            forget_clabels = clabels[forget_mask]
            loss_forget = criterion(forget_logits, forget_clabels) if forget_mask.any() else 0

            loss = loss_retain - beta * loss_forget

        else:  # (images, clabels) 格式
            images, clabels = batch
            images, clabels = images.to(device), clabels.to(device)
            out = model(images)
            loss = criterion(out, clabels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 应用梯度掩码（如果有）
        if mask:
            for name, param in model.named_parameters():
                if param.grad is not None and name in mask:
                    param.grad *= mask[name]

        optimizer.step()

        # 定期执行神经元重置
        if (batch_i + 1) % reset_interval == 0:
            # 执行神经元重置
            reset_stats = cb_engine.step(activation_hook.activations)
            total_reset += reset_stats['total_reset']

            # 计算有效秩（使用最后一层的激活）
            if activation_hook.activations:
                last_layer = layer_names[-1] if layer_names else None
                if last_layer and last_layer in activation_hook.activations:
                    eff_rank = cb_engine.compute_effective_rank(
                        activation_hook.activations[last_layer]
                    )
                    effective_ranks.append(eff_rank)

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'reset': total_reset,
            'eff_rank': f'{effective_ranks[-1]:.2f}' if effective_ranks else 'N/A'
        })

    # 清理钩子
    activation_hook.remove()

    # 调度学习率
    if lr_schedule is not None:
        lr_schedule.step()

    print(f'\nContinual Backprop: Total neurons reset: {total_reset}')
    if effective_ranks:
        print(f'Average effective rank: {np.mean(effective_ranks):.4f}')

    return cb_engine


def fit_one_cycle_with_cb(
    train_loader, model, optimizer, criterion, scheduler, unlearn_method, mask, beta=0.15, args={}, cb_engine=None
):
    """
    带有 Continual Backprop 的训练循环

    这是 fit_one_cycle 的增强版本，在每个训练步骤后执行神经元重置

    Args:
        train_loader: 训练数据加载器
        model: 神经网络模型
        optimizer: 优化器
        criterion: 损失函数
        scheduler: 学习率调度器
        unlearn_method: 遗忘方法名称
        mask: 梯度掩码
        beta: 遗忘损失权重
        args: 配置参数
        cb_engine: ContinualBackpropEngine 实例

    Returns:
        beta: 返回 beta 值
        cb_engine: 更新后的 ContinualBackpropEngine 实例
    """
    device = next(model.parameters()).device
    model.train()

    # 创建或使用已有的 Continual Backprop 引擎
    if cb_engine is None:
        cb_engine = create_continual_backprop_engine(model, args)

    # 设置激活值捕获
    layer_names = list(cb_engine.neuron_info.keys())
    activation_hook = ActivationHook(model, layer_names)

    reset_interval = getattr(args, 'cb_reset_interval', 10)
    total_reset = 0

    pbar = tqdm(train_loader, total=len(train_loader))
    for batch_i, batch in enumerate(pbar):
        activation_hook.clear()

        if unlearn_method in ['ga_plus', 'salun', 'randomlabel', 'continual_backprop']:
            loss_retain, loss_forget = training_step_split(model, batch, criterion, beta, unlearn_method)
            loss = loss_retain + loss_forget
        else:
            loss = training_step(model, batch, criterion)
            loss = -loss if unlearn_method == 'ga' else loss

        optimizer.zero_grad()
        loss.backward()

        if mask and unlearn_method == 'salun':
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]

        optimizer.step()

        # 神经元重置
        if (batch_i + 1) % reset_interval == 0:
            reset_stats = cb_engine.step(activation_hook.activations)
            total_reset += reset_stats['total_reset']

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'reset': total_reset})

    activation_hook.remove()
    scheduler.step()

    print(f'Continual Backprop: Total neurons reset this epoch: {total_reset}')
    return beta, cb_engine

