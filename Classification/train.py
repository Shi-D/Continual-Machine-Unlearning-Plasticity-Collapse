import os
import sys
import time
import torch
import torch.nn.functional as F
import json
import torch.nn.parallel

import random
import data
import models
import utils
import wandb
import numpy as np
from arg_parser import parse_args
from datetime import datetime
from torch.utils.data import DataLoader
from methods.PCU import PCU_Engine

import yaml
from methods.ContinualBackprop import ContinualBackpropEngine, create_continual_backprop_engine
from torch.utils.data import Dataset, Subset, random_split, ConcatDataset



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_train_dl(retain_dl, forget_dl, args, num_classes=10, loaders=None):
    if args.unlearn_method in ['ga_plus', 'continual_backprop']:
        unlearning_data = utils.LossData2(forget_data=forget_dl.dataset, retain_data=retain_dl.dataset)
        train_loader = DataLoader(
            unlearning_data, batch_size=args.batch_size, shuffle=True, pin_memory=True
        )
    elif args.unlearn_method in ['randomlabel', 'salun']:
        unlearning_data = utils.RandomData(forget_data=forget_dl.dataset,
                                           retain_data=retain_dl.dataset, num_classes=num_classes)

        train_loader = DataLoader(
            unlearning_data, batch_size=args.batch_size, shuffle=True, pin_memory=True
        )
    elif args.unlearn_method in ['retrain', 'finetune']:
        train_loader = retain_dl
    elif args.unlearn_method in ['ga']:
        train_loader = forget_dl
    elif args.unlearn_method in ['sfron']:
        train_loader = loaders
    else:
        train_loader = None

    return train_loader


def init_opt(args):
    criterion = F.cross_entropy
    optimizer = torch.optim.SGD(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd if args.curve is None else 0.0
    )

    if args.milestones:
        milestones = np.array(args.milestones.split(',')).astype(int)
        print('===milestones===', milestones)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1, last_epoch=-1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    return optimizer, criterion, scheduler


def train_per_step(model, num_classes, loader, forget_step, dir, pcu_engine, args, use_PCU=False, cb_engine=None):

    args.forget_step = forget_step
    if args.unlearn_method in ['delete', 'munba']:
        model = utils.train(loader, model, None, None, None, num_classes, mask=None, epoch=0, args=args)
        return model, None, {}, [], None, cb_engine

    optimizer, criterion, scheduler = init_opt(args)
    acc_results = {}
    training_time_sum = 0
    epoch_ER_list = []

    mask = None
    if args.unlearn_method == 'salun':
        step_mask_s = time.time()
        mask = utils.generate_mask(model, optimizer, criterion, loader['forget_dl'])

    # Initialize Continual Backprop
    use_continual_backprop = args.unlearn_method == 'continual_backprop' or getattr(args, 'use_continual_backprop', False)
    if use_continual_backprop and cb_engine is None:
        cb_engine = create_continual_backprop_engine(model, args)
        print(f'Initialized ContinualBackpropEngine with reset_ratio={getattr(args, "cb_reset_ratio", 0.001)}')

    pcu_res = None
    for epoch in range(args.epochs):
        time_ep = time.time()

        if args.unlearn_method in ['sfron']:
            utils.train(loader, model, optimizer, criterion, scheduler, num_classes, mask=mask,
                                    args=args)
        elif args.unlearn_method == 'continual_backprop':
            # Continual Backprop
            import unlearn
            cb_engine = unlearn.continual_backprop(
                loader['train_loader'], model, optimizer, criterion, scheduler,
                mask=mask, args=args, cb_engine=cb_engine
            )
        else:
            utils.train(loader['train_loader'], model, optimizer, criterion, scheduler, num_classes,
                                    mask=mask,
                                    args=args)

            if use_continual_backprop and args.unlearn_method != 'continual_backprop':
                reset_stats = cb_engine.step()
                if reset_stats['total_reset'] > 0:
                    print(f'  Epoch {epoch}: Reset {reset_stats["total_reset"]} neurons')

        time_test = time.time()
        training_time = time_test - time_ep
        training_time_sum += training_time

        train_remain_acc, train_retain_acc, train_forget_acc, test_retain_acc = 0, 0, 0, 0
        log_res = {'train_remain_acc': train_remain_acc,
                   'train_retain_acc': train_retain_acc,
                   'train_forget_acc': train_forget_acc,
                   'test_retain_acc': test_retain_acc,
                   "lr": optimizer.param_groups[0]["lr"],
                   "train_time": training_time, "test_time": time.time() - time_test}

        # Caculate ER after each epoch
        if use_PCU and pcu_engine is not None:
            pcu_res = pcu_engine.project(model)
            epoch_ER = pcu_res['ER']
            epoch_ER_list.append(epoch_ER)
        print(log_res)
        wandb.log(log_res)

    return model, optimizer, acc_results, epoch_ER_list, pcu_res, cb_engine


if __name__ == '__main__':

    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.unlearn_method == 'finetune2':
        dir = f'{args.dir}{args.unlearn_method}/seed{args.seed}/'
    else:
        dir = f'{args.dir}{args.unlearn_method}/seed{args.seed}_beta{args.beta}_lr{args.lr}/'

    print('\n=====save_dir=====', dir, '\n')
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

    with open(os.path.join(dir, 'command.sh'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')

    set_seed(args.seed)

    loaders, num_classes = data.loaders(
        args.dataset,
        args.unlearn_type,
        args.forget_ratio,
        args.data_path,
        args.batch_size,
        args.num_workers,
        args.transform,
        args.use_test
    )

    architecture = getattr(models, args.model)

    model = architecture.base(num_classes=num_classes, **architecture.kwargs)

    if args.original_pth is not None:
        checkpoint_original = torch.load(args.original_pth)

        new_state_dict = {}
        for v_i, v in checkpoint_original['model_state'].items():
            new_key = v_i.replace("module.", "")
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)

    if args.mask_path is not None:
        mask = torch.load(args.mask_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    proj_name = 'Plasticity_{}_{}_{}_{}{}'.format(args.model, args.dataset, args.unlearn_method, args.unlearn_type,
                                                  int(args.forget_ratio))
    watermark = "s{}_lr{}_b{}".format(args.seed, args.lr, args.beta)
    print('\nproj_name', proj_name)
    print('watermark', watermark, '\n')

    wandb.init(project=proj_name, name=watermark)
    wandb.config.update(args)
    wandb.watch(model)

    has_bn = utils.check_bn(model)
    test_res = {'loss': None, 'accuracy': None, 'nll': None}
    training_time_sum = 0
    train_retain_acc, train_forget_acc, test_retain_acc, test_forget_acc = 0, 0, 0, 0

    remain_dl = loaders['train']
    test_dl = loaders['test']
    forget_dl = None
    forget_class_list = config['forget_list'] if args.unlearn_type == 'class' else None
    print('==' * 20)
    print('forget_class_list', forget_class_list)
    print('==' * 20)

    forget_num = int(len(remain_dl.dataset) * args.forget_ratio) if args.unlearn_type == 'random' else int(
        args.forget_ratio)
    forgotten_dl_list = []

    use_PCU = getattr(args, 'use_PCU', False)


    pcu_engine = None
    adaptive_results = {}

    if use_PCU:
        pcu_engine = PCU_Engine(r=3, max_ws=10, min_ws=3, significance_level=0.05, layer_name=args.layer_name)

    # For Continual Backprop
    cb_engine = None
    use_continual_backprop = args.unlearn_method == 'continual_backprop' or getattr(args, 'use_continual_backprop', False)
    if use_continual_backprop:
        cb_engine = create_continual_backprop_engine(model, args)
        print(f'Initialized ContinualBackpropEngine:')
        print(f'  - reset_ratio: {getattr(args, "cb_reset_ratio", 0.001)}')
        print(f'  - maturity_threshold: {getattr(args, "cb_maturity_threshold", 1000)}')
        print(f'  - utility_decay: {getattr(args, "cb_utility_decay", 0.99)}')

    for forget_step in range(args.forget_times):
        if use_PCU and pcu_engine is not None:
            pcu_engine.save_params_before_task(model)

        args.forget_step = forget_step

        all_task_s = time.time()
        print('============forget_step', forget_step)

        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        dir_sub = dir

        '''
        ===============================
        Data Loader
        ===============================
        '''
        step_data_s = time.time()
        remain_dl, retain_dl, forget_dl = utils.update_dl(remain_dl, forget_step, forget_num, forget_class_list,
                                                          args.unlearn_type, num_classes, args)
        if forget_dl is not None:
            forgotten_dl_list.append(forget_dl)
        train_loader = init_train_dl(retain_dl, forget_dl, args, num_classes=num_classes, loaders=loaders)

        if args.unlearn_type == 'class':
            test_dl = utils.update_test_dl(test_dl, forget_step, forget_num, forget_class_list)
            print(
                f'forget_class: {forget_class_list[forget_step]}, forget_dl: {len(forget_dl)}, remain_dl: {len(remain_dl)}, retain_dl: {len(retain_dl)}, test_dl: {len(test_dl)}')

        loader = {'remain_dl': remain_dl, 'retain_dl': retain_dl, 'forget_dl': forget_dl, 'test_dl': test_dl,
                  'train_loader': train_loader}

        '''
        ===============================
        Training
        ===============================
        '''
        step_time_s = time.time()
        model, optimizer, acc_results, epoch_ER_list, pcu_res, cb_engine = train_per_step(
            model, num_classes, loader, forget_step, dir_sub, pcu_engine, args, use_PCU, cb_engine
        )

        '''
        ===============================
        Evaluation Accuracy
        ===============================
        '''
        eval_time_s = time.time()

        train_remain_acc, train_retain_acc = 0, 0
        train_remain_acc = utils.evaluate_acc(model, loader['remain_dl'], device)  # remain data is all remaining data
        train_retain_acc = utils.evaluate_acc(model, loader['retain_dl'], device)  # retain data is used as retain data in unlearning process
        train_forget_acc = utils.evaluate_acc(model, loader['forget_dl'], device)
        test_retain_acc = utils.evaluate_acc(model, loader['test_dl'], device)
        acc_results['end'] = {'train_remain_acc': train_remain_acc,
                              'train_retain_acc': train_retain_acc,
                              'train_forget_acc': train_forget_acc,
                              'test_retain_acc': test_retain_acc}
        print(acc_results['end'])

        eval_time_m = time.time()

        # History forgotten data
        forgotten_accs = []
        for forgotten_dl in forgotten_dl_list:
            forgotten_acc = utils.evaluate_acc(model, forgotten_dl, device)
            forgotten_accs.append(forgotten_acc)
        acc_results['forgotten_accs'] = forgotten_accs

        test_len = len(loader['test_dl'].dataset)
        train_len = len(loader['remain_dl'].dataset)
        print('train_len', train_len, 'test_len', test_len)
        min_len = min(test_len, train_len)

        shadow_train_ds, _ = random_split(loader['remain_dl'].dataset, [min_len, train_len-min_len])
        shadow_train_loader = torch.utils.data.DataLoader(shadow_train_ds, batch_size=args.batch_size, shuffle=False)
        shadow_test_loader = loader['test_dl']
        target_train_forget = loader['forget_dl']


        '''
        ===============================
        Save File and Checkpoints
        ===============================
        '''
        step_save_s = time.time()
        utils.save_checkpoint(
            dir_sub,
            args.epochs,
            name=f'checkpoint_seed{args.seed}_step{forget_step}_end',
            model_state=model.state_dict()
        )

        dir_json = dir_sub.replace('results', 'results_json')
        print('dir_json:', dir_json)
        os.makedirs(dir_json, exist_ok=True)

        save_path = os.path.join(
            dir_json,
            f'results_seed{args.seed}_step{forget_step}_end.json'
        )

        with open(save_path, 'w') as f:
            json.dump(acc_results, f, indent=2)

        print('++' * 20)
        print('One save time:', time.time() - step_save_s)
        print('++' * 20)

        print('++' * 20)
        print('One TASK ALL time', time.time() - all_task_s)
        print('++' * 20)

        '''
        ===============================
        PCU Computation
        ===============================
        '''
        failure_mode_acc = None
        failure_mode_ER = None
        failure_mode_CO = None
        acc_stats = {}
        ER_stats = {}
        CO_stats = {}
        task_ER = -int('inf')
        task_CO = -int('inf')

        if use_PCU and pcu_engine is not None:
            # 1. task update（for subspace W update）
            pcu_engine.record_task_update(model)
            pcu_engine._incremental_update()

            # 2. get ER
            if pcu_res is not None:
                task_ER = pcu_res['ER']
                task_CO = pcu_res['CO'][0] if pcu_res['CO'] else -int('inf')

            # 3. update failure modes detection window
            pcu_engine.update_window_acc(train_forget_acc)
            pcu_engine.update_window_ER(task_ER)
            pcu_engine.update_window_CO(task_CO)

            # 4. get failure mode both from acc and ER/CO
            failure_mode_acc, acc_stats = pcu_engine.get_failure_mode_acc_c()
            failure_mode_ER, ER_stats = pcu_engine.get_failure_mode_ER_c()
            failure_mode_CO, CO_stats = pcu_engine.get_failure_mode_CO()

            # 5. save results
            adaptive_results[forget_step] = {
                'forget_acc': train_forget_acc,
                'task_CO': task_CO,
                'task_ER': task_ER,
                'failure_mode_acc': failure_mode_acc,
                'failure_mode_ER': failure_mode_ER,
                'failure_mode_CO': failure_mode_CO,
                'acc_stats': acc_stats,
                'ER_stats': ER_stats,
                'CO_stats': CO_stats,
                'epoch_ER_list': epoch_ER_list
            }


        '''
        ===============================
        Save Adaptive Results to JSON
        ===============================
        '''
        if use_PCU:  # use_adaptive_lambda
            adaptive_json_path = os.path.join(dir_sub, f'PCU_results_seed{args.seed}.json')
            def convert_to_serializable(obj):
                if obj is None:
                    return None
                elif isinstance(obj, torch.Tensor):
                    return obj.detach().cpu().numpy().tolist()
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(i) for i in obj]
                return obj

            with open(adaptive_json_path, 'w') as f:
                json.dump(convert_to_serializable(adaptive_results), f, indent=2)
            print(f'Adaptive results saved to {adaptive_json_path}')
