import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import curves
from torch.utils.data import DataLoader, Subset, dataset, Dataset, random_split
import torch
import numpy as np
from tqdm import tqdm
import unlearn
import time

def get_representation(model, dataloader, arch, device="cuda", n_components=50):
    """
    通用特征提取函数：支持 PreResNetBase 和 VGGBase
    同步处理 Shuffle 标签，并进行 PCA 降维以节省空间。
    """
    model.eval()
    model.to(device)

    raw_features = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"Extracting {arch} Features"):
            inputs = inputs.to(device)

            # --- 逻辑区分：PreResNet ---
            if 'resnet' in arch.lower():
                # 按照 PreResNetBase 的 forward 流程执行，但在 fc 之前停止
                x = model.conv1(inputs)
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.bn(x)
                x = model.relu(x)
                x = model.avgpool(x)
                features = x.view(x.size(0), -1)  # 展平后的 Penultimate feature

            # --- 逻辑区分：VGG ---
            elif 'vgg' in arch.lower():
                # 按照 VGGBase 的 forward 流程执行
                x = inputs
                for layers, activations, pooling in zip(model.layer_blocks,
                                                        model.activation_blocks,
                                                        model.poolings):
                    for layer, activation in zip(layers, activations):
                        x = activation(layer(x))
                    x = pooling(x)
                x = x.view(x.size(0), -1)
                # 提取进入最后一个 Linear 层前的特征 (classifier 索引 0-5)
                # 对应 [Dropout, Linear, ReLU, Dropout, Linear, ReLU]
                features = model.classifier[:6](x)

            else:
                raise ValueError(f"Unsupported architecture: {arch}")

            raw_features.append(features.detach().cpu().numpy())
            # 即使 dataloader 是 shuffle 的，labels 也会按顺序与 features 对应
            all_labels.append(labels.numpy())

    # 合并为大矩阵
    features_concat = np.concatenate(raw_features, axis=0)
    labels_concat = np.concatenate(all_labels, axis=0)

    # # PCA 降维：将高维特征压缩以节省存储空间并加速后续 t-SNE [cite: 78, 79]
    # print(f"Applying PCA: {features_concat.shape[1]}D -> {n_components}D")
    # pca = PCA(n_components=n_components)
    # reduced_features = pca.fit_transform(features_concat)

    return features_concat, labels_concat

def l2_regularizer(weight_decay):
    def regularizer(model):
        l2 = 0.0
        for p in model.parameters():
            l2 += torch.sqrt(torch.sum(p ** 2))
        return 0.5 * weight_decay * l2
    return regularizer


def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)
    return schedule


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, f'{name}.pt')
    print('checkpoint filepath', filepath)
    torch.save(state, filepath)


def train(loader, model, optimizer, criterion, regularizer=None, num_classes=10, mask=None, epoch=1, args={}):
    kwargs = {"num_classes": num_classes, "retain_ratio": args.retain_ratio, "epoch": epoch}
    res = getattr(unlearn, args.unlearn_method)(loader, model, optimizer, criterion, regularizer, mask, args, **kwargs)

    return res


def test(test_loader, model, criterion, regularizer=None, **kwargs):
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0

    model.eval()
    with torch.no_grad():
        for input, target in test_loader:
            input = input.cuda()
            target = target.cuda()

            output = model(input, **kwargs)
            nll = criterion(output, target)
            loss = nll.clone()
            if regularizer is not None:
                loss += regularizer(model)

            nll_sum += nll.item() * input.size(0)
            loss_sum += loss.item() * input.size(0)
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'nll': nll_sum / len(test_loader.dataset),
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }



def predictions(test_loader, model, **kwargs):
    model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        input = input.cuda()
        output = model(input, **kwargs)
        probs = F.softmax(output, dim=1)
        preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def isbatchnorm(module):
    return issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm) or \
           issubclass(module.__class__, curves._BatchNorm)


def _check_bn(module, flag):
    if isbatchnorm(module):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if isbatchnorm(module):
        module.reset_running_stats()


def _get_momenta(module, momenta):
    if isbatchnorm(module):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if isbatchnorm(module):
        module.momentum = momenta[module]


def update_bn(loader, model, **kwargs):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    num_samples = 0
    for input, _ in loader:
        input = input.cuda()
        batch_size = input.data.size(0)

        momentum = batch_size / (num_samples + batch_size)
        for module in momenta.keys():
            module.momentum = momentum

        model(input, **kwargs)
        num_samples += batch_size

    model.apply(lambda module: _set_momenta(module, momenta))


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item(), len(preds), torch.tensor(torch.sum(preds == labels).item() / len(preds)) * 100

def evaluate_acc_batch(model, batch, device):
    images, clabels = batch
    images, clabels = images.to(device), clabels.to(device)
    out = model(images)
    return accuracy(out, clabels)

def evaluate_acc(model, val_loader, device):
    if not val_loader:
        return 0

    model.eval()
    corr, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            corr_, total_, _ = evaluate_acc_batch(model, batch, device)
            corr += corr_
            total += total_
    torch.cuda.empty_cache()
    return corr/total


def evaluate_acc_loss(model, loader, device, criterion=torch.nn.CrossEntropyLoss()):
    if not loader:
        return 0.0, 0.0

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    avg_acc = correct / total

    return avg_acc, avg_loss

"""for SCRUB: imported from https://github.com/HobbitLong/RepDistiller"""
class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LossData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.data = [(x, y, 1) for x, y in forget_data] + [(x, y, 0) for x, y in retain_data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class LossData2(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if index < self.forget_len:  # forget
            x, y = self.forget_data[index]
            return x, y, 1
        else:  # retain
            x, y = self.retain_data[index - self.forget_len]
            return x, y, 0

class RandomData(Dataset):
    def __init__(self, forget_data, retain_data, num_classes):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.num_classes = num_classes
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if index < self.forget_len:  # forget
            x, y = self.forget_data[index]

            unlearninglabels = list(range(self.num_classes))
            rnd = random.choice(unlearninglabels)
            while rnd == y:
                rnd = random.choice(unlearninglabels)
            return x, rnd, 1
        else:  # retain
            x, y = self.retain_data[index - self.forget_len]
            return x, y, 0


####### SCRUB ######
class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, gamma, beta, split,
                  print_freq=12, quiet=False):
    """One epoch distillation"""
    # set modules as train()
    # for module in module_list:
    #     module.train()
    # # set teacher as eval()
    # module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]
    model_s.train()
    model_t.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    kd_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    acc_max_top1 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):

        input = input.cuda()
        target = target.cuda()
        data_time.update(time.time() - end)

        input = torch.Tensor(input).float()
        # target = torch.squeeze(torch.Tensor(target).long())

        # ===================forward=====================
        logit_s = model_s(input)
        with torch.no_grad():
            logit_t = model_t(input)

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)
        loss_kd = 0

        if split == "minimize":
            loss = gamma * loss_cls + beta * loss_div
            # loss = gamma * loss_cls + alpha * loss_div + beta * loss_kd

        elif split == "maximize":
            loss = -loss_div

        if split == "minimize" and not quiet:
            acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))
        elif split == "maximize" and not quiet:
            kd_losses.update(loss.item(), input.size(0))
            acc_max, _ = accuracy(logit_s, target, topk=(1, 5))
            acc_max_top1.update(acc_max.item(), input.size(0))


    # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

    if split == "maximize":
        if not quiet:
            # if idx % print_freq == 0:
            print('*** Maximize step ***')
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Forget_Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=kd_losses, top1=acc_max_top1))
            # sys.stdout.flush()
    elif split == "minimize":
        if not quiet:
            print('*** Minimize step ***')
            # print(' * Acc@1 {top1.avg:.3f} '.format(top1=top1))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Retain_Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

        return top1.avg, losses.avg
    else:
        # module_list[0] = model_s
        # module_list[-1] = model_t
        return kd_losses.avg

'''salun generate mask'''
def generate_mask(model, optimizer, criterion, forget_dl):
    print('Generate mask for salun...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gradients = {}
    model.eval()

    for name, param in model.named_parameters():
        gradients[name] = torch.zeros_like(param)

    for i, (image, target) in enumerate(forget_dl):
        image = image.to(device)
        target = target.to(device)

        # compute output
        output_clean = model(image)
        loss = - criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

    threshold = 0.5

    sorted_dict_positions = {}
    hard_dict = {}

    # Concatenate all tensors into a single tensor
    all_elements = - torch.cat([tensor.flatten() for tensor in gradients.values()])

    # Calculate the threshold index for the top threshold% elements
    threshold_index = int(len(all_elements) * threshold)

    # Calculate positions of all elements
    positions = torch.argsort(all_elements)
    ranks = torch.argsort(positions)

    start_index = 0
    for key, tensor in gradients.items():
        num_elements = tensor.numel()  # len
        # tensor_positions = positions[start_index: start_index + num_elements]
        tensor_ranks = ranks[start_index : start_index + num_elements]

        sorted_positions = tensor_ranks.reshape(tensor.shape)
        sorted_dict_positions[key] = sorted_positions

        # Set the corresponding elements to 1
        threshold_tensor = torch.zeros_like(tensor_ranks)
        threshold_tensor[tensor_ranks < threshold_index] = 1
        threshold_tensor = threshold_tensor.reshape(tensor.shape)
        hard_dict[key] = threshold_tensor
        start_index += num_elements
        # if 'weight' in key and tensor.dim() >= 2:
        #     print('name:', key, 'hard_dict[key]:', hard_dict[key])


    return hard_dict

def get_labels(ds):
    if isinstance(ds, Subset):
        base_ds = ds.dataset
        if hasattr(base_ds, 'targets'):
            base_labels = base_ds.targets
        elif hasattr(base_ds, 'samples'):  # ImageFolder
            base_labels = [s[1] for s in base_ds.samples]
        else:
            raise ValueError("Base dataset has no targets or samples")
        # Subset 内的 labels
        return [base_labels[i] for i in ds.indices]
    else:
        if hasattr(ds, 'targets'):
            return ds.targets
        elif hasattr(ds, 'samples'):
            return [s[1] for s in ds.samples]
        else:
            raise ValueError("Dataset has no targets or samples")


import random
import torch
from torch.utils.data import DataLoader, Subset, random_split

def update_dl(remain_dl, forget_step, samples, forget_class_list, unlearn_type, num_classes, args):
    remain_ds = remain_dl.dataset
    indices = None
    while isinstance(remain_ds, Subset):
        if indices is None:
            indices = remain_ds.indices
        else:
            indices = [remain_ds.indices[i] for i in indices]
        remain_ds = remain_ds.dataset

    original_ds = remain_ds
    if indices is None:
        indices = list(range(len(original_ds)))  # 初始全部样本

    # 获取 labels
    if hasattr(original_ds, 'targets'):
        labels_all = original_ds.targets
    elif hasattr(original_ds, 'samples'):
        labels_all = [s[1] for s in original_ds.samples]
    else:
        raise ValueError("Base dataset has no targets or samples")

    # 当前 remain 的 labels
    labels = [labels_all[i] for i in indices]

    if unlearn_type == 'random':
        remain_size = len(indices) - samples
        remain_indices_new, forget_indices = random_split(indices, [remain_size, samples])
        retain_indices, _ = random_split(remain_indices_new, [samples*5, remain_size - samples*5])

    elif unlearn_type == 'class':
        # 确定 forget classes
        forget_classes = [forget_class_list[forget_step]]
        print('forget_classes', forget_classes)

        forget_indices = [i for i, y in zip(indices, labels) if y in forget_classes]
        remain_indices_new = [i for i, y in zip(indices, labels) if y not in forget_classes]
        retain_indices = random.sample(remain_indices_new, len(forget_indices)*5)

        print('len(forget_classes), len(retain_indices), len(remain_indices_new)',
              len(forget_classes), len(retain_indices), len(remain_indices_new))
    else:
        raise ValueError(f"Unknown unlearn_type: {unlearn_type}")

    # 构造 Subset
    forget_ds = Subset(original_ds, forget_indices)
    retain_ds = Subset(original_ds, retain_indices)
    remain_ds_new = Subset(original_ds, remain_indices_new)

    # DataLoader
    forget_dl = DataLoader(
        forget_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    retain_dl = DataLoader(
        retain_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    remain_dl_new = DataLoader(
        remain_ds_new,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    if args.unlearn_method == 'finetune':
        print('args.unlearn_method == finetune', args.unlearn_method == 'finetune')
        retain_dl = remain_dl_new
    return remain_dl_new, retain_dl, forget_dl


def update_test_dl(test_dl, forget_step, forget_num, forget_class_list):
    forget_classes = [forget_class_list[forget_step]]
    test_ds = test_dl.dataset
    indices = None
    while isinstance(test_ds, Subset):
        if indices is None:
            indices = test_ds.indices
        else:
            indices = [test_ds.indices[i] for i in indices]
        test_ds = test_ds.dataset

    original_ds = test_ds  # 永远指向原始 dataset
    if indices is None:
        indices = list(range(len(original_ds)))  # 初始全部样本

    # 获取 labels
    if hasattr(original_ds, 'targets'):
        labels_all = original_ds.targets
    elif hasattr(original_ds, 'samples'):
        labels_all = [s[1] for s in original_ds.samples]
    else:
        raise ValueError("Base dataset has no targets or samples")

    # 当前 remain 的 labels
    labels = [labels_all[i] for i in indices]

    test_indices_new = [i for i, y in zip(indices, labels) if y not in forget_classes]

    # 构造 Subset
    test_ds_new = Subset(original_ds, test_indices_new)
    test_dl_new = DataLoader(
        test_ds_new,
        batch_size=test_dl.batch_size,
        shuffle=False,
        num_workers=test_dl.num_workers,
        pin_memory=True
    )
    return test_dl_new

