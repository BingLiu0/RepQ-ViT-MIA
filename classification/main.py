import argparse
import os
import random
import torch
import torch.nn as nn
import numpy as np
import shutil
import datetime
import torch.optim as optim
from torch.optim import lr_scheduler
from tools import *
from quant import *
from sklearn.metrics import accuracy_score, precision_score, recall_score
from utils import MyConfig


def get_args_parser():
    parser = argparse.ArgumentParser(description="RepQ-ViT", add_help=False)
    parser.add_argument("--model", default="vit_small", type=str,
                        choices=['vit_small', 'vit_base','deit_tiny', 'deit_small', 'deit_base', 'swin_tiny', 'swin_small'])
    parser.add_argument('--dataset', default="CIFAR10", type=str,
                        choices=['CIFAR10', 'CIFAR100','ImageNet100', 'ISIC2018'])
    parser.add_argument('--attack_method_attention', default="rollout", type=str,
                        choices=['rollout', 'last_attention', 'EncoderMI'])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--w_bits', default=8, type=int)
    parser.add_argument('--a_bits', default=8, type=int)
    parser.add_argument('--noise_repeat', default=10, type=int)
    parser.add_argument('--metric', default="cos-sim", type=str, 
                        choices=['cos-sim', 'pearson', 'Euclid', 'CE'])
    parser.add_argument('--head_fusion', default='mean', type=str)
    parser.add_argument('--discard_ratio', default=0, type=float)
    return parser


def attack_eval(path, attack_model, test_loader_attack, device, method):
    label, pred, soft_pred = get_data_for_attack_eval(attack_model, test_loader_attack, device)
    precision = precision_score(y_true=label, y_pred=pred, average = "macro", zero_division=1)
    recall = recall_score(y_true=label, y_pred=pred, average = "macro")
    accuracy = accuracy_score(y_true=label, y_pred=pred)
    np.save(path + f"/res_precision_{method}_classifier.npy", precision)
    np.save(path + f"/res_recall_{method}_classifier.npy", recall)
    np.save(path + f"/res_accuracy_{method}_classifier.npy", accuracy)
    np.save(path + f"/res_label_attack_{method}_classifier.npy", label)
    np.save(path + f"/res_soft_pred_attack_{method}_classifier.npy", soft_pred)


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    print(args)
    seed(args.seed)
    save_folder = './saved_models'
    os.makedirs(save_folder, exist_ok=True)
    
    config_dict = {'CIFAR10': "config/CIFAR10/",
                'CIFAR100': "config/CIFAR100/",
                'ImageNet100': "config/ImageNet100/",
                'ISIC2018': "config/ISIC2018/",
                }
    
    config = MyConfig.MyConfig(path=config_dict[args.dataset])

    now = str(datetime.datetime.now())[:19]
    now = now.replace(":","_")
    now = now.replace("-","_")
    now = now.replace(" ","_")

    src_dir = config.path.config_path
    path = config.path.result_path + args.dataset + "/" + args.model + "_" + str(now)
    os.mkdir(path)
    dst_dir = path+ "/config.yaml"
    shutil.copy(src_dir,dst_dir)

    model_zoo = {
        'vit_small' : 'vit_small_patch16_224',
        'vit_base' : 'vit_base_patch16_224',

        'deit_tiny' : 'deit_tiny_patch16_224',
        'deit_small': 'deit_small_patch16_224',
        'deit_base' : 'deit_base_patch16_224',

        'swin_tiny' : 'swin_tiny_patch4_window7_224',
        'swin_small': 'swin_small_patch4_window7_224',
    }
    
    device = torch.device(args.device)
    
    print(f'Performing experiments for {args.model} on {args.dataset} ....................................................................................')
    
    print('Building dataloader ....................................')
    dataloaders_target, dataset_sizes_target, dataloaders_shadow, dataset_sizes_shadow = build_dataset(args, config)
    
    criterion = nn.CrossEntropyLoss()
    
    print('Building target model .........................................')
    target_model = build_model(model_zoo[args.model], config.general.num_classes)
    target_model.to(device)
    optimizer = optim.SGD(target_model.parameters(), lr=config.vit_learning.learning_rate, momentum=config.vit_learning.momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config.vit_learning.decrease_lr_factor, gamma=config.vit_learning.decrease_lr_every)
    target_model, epoch_loss_acc_target = train_model(target_model, criterion, optimizer, exp_lr_scheduler, dataloaders_target, dataset_sizes_target, num_epochs=config.vit_learning.epochs)
    save_path = os.path.join(save_folder, f'target_{args.model}_{args.dataset}_{str(now)}.pth')
    torch.save(target_model.state_dict(), save_path)
    np.save(path + "/res_epoch_loss_acc_target"+".npy", epoch_loss_acc_target)
    target_train_test_accuracy = train_test_acc(target_model, dataloaders_target, dataset_sizes_target, device)
    np.save(path + "/res_target_train_test_accuracy"+".npy", target_train_test_accuracy)
    
    print('Building shadow model .........................................')
    shadow_model = build_model(model_zoo[args.model], config.general.num_classes)
    shadow_model.to(device)
    optimizer = optim.SGD(shadow_model.parameters(), lr=config.vit_learning.learning_rate, momentum=config.vit_learning.momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config.vit_learning.decrease_lr_factor, gamma=config.vit_learning.decrease_lr_every)
    shadow_model, epoch_loss_acc_shadow = train_model(shadow_model, criterion, optimizer, exp_lr_scheduler, dataloaders_shadow, dataset_sizes_shadow, num_epochs=config.vit_learning.epochs)
    save_path = os.path.join(save_folder, f'shadow_{args.model}_{args.dataset}_{str(now)}.pth')
    torch.save(shadow_model.state_dict(), save_path)
    np.save(path + "/res_epoch_loss_acc_shadow"+".npy", epoch_loss_acc_shadow)
    shadow_train_test_accuracy = train_test_acc(shadow_model, dataloaders_shadow, dataset_sizes_shadow, device)
    np.save(path + "/res_shadow_train_test_accuracy"+".npy", shadow_train_test_accuracy)
    
    print('Performing membership inference attack ..........................')
    # attack_model, test_loader_attack = mia_base(args, config, target_model, dataloaders_target, shadow_model, dataloaders_shadow)
    # attack_eval(path, attack_model, test_loader_attack, args.device, method="mia_base")
    
    # attack_model, test_loader = mia_attention_classifier(args, config, target_model, dataloaders_target, shadow_model, dataloaders_shadow)
    # attack_eval(path, attack_model, test_loader, args.device, method=args.attack_method_attention)
     
    mia_attention_threshold(args, config, path, target_model, dataloaders_target, shadow_model, dataloaders_shadow)
    
    print('Performing quantization .......................................')
    for data, _ in dataloaders_target["test"]:
        calib_target_data = data.to(device)
        break
    calib_target_data.to(device)
    
    for data, _ in dataloaders_shadow["test"]:
        calib_shadow_data = data.to(device)
        break
    calib_shadow_data.to(device)

    wq_params = {'n_bits': args.w_bits, 'channel_wise': True}
    aq_params = {'n_bits': args.a_bits, 'channel_wise': False}
    
    q_target_model = quant_model(target_model, input_quant_params=aq_params, weight_quant_params=wq_params)
    q_target_model.to(device)
    q_target_model.eval()
    
    set_quant_state(q_target_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        _ = q_target_model(calib_target_data)
        
    scale_reparam(args, q_target_model)
    
    set_quant_state(q_target_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        _ = q_target_model(calib_target_data)
    
    save_path = os.path.join(save_folder, f'q_target_{args.model}_{args.dataset}_{str(now)}.pth')
    torch.save(q_target_model.state_dict(), save_path)
    
    q_shadow_model = quant_model(shadow_model, input_quant_params=aq_params, weight_quant_params=wq_params)
    q_shadow_model.to(device)
    q_shadow_model.eval()

    set_quant_state(q_shadow_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        _ = q_shadow_model(calib_shadow_data)
        
    scale_reparam(args, q_shadow_model)
    
    set_quant_state(q_shadow_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        _ = q_shadow_model(calib_shadow_data)

    save_path = os.path.join(save_folder, f'q_shadow_{args.model}_{args.dataset}_{str(now)}.pth')
    torch.save(q_shadow_model.state_dict(), save_path)
    
    print('Performing quantized membership inference attack ..........................')
    # attack_model, test_loader_attack = mia_base(args, config, q_target_model, dataloaders_target, q_shadow_model, dataloaders_shadow)
    # attack_eval(attack_model, test_loader_attack, args.device, method="mia_base_quantized")
    
    # attack_model, test_loader = mia_attention_classifier(args, config, q_target_model, dataloaders_target, q_shadow_model, dataloaders_shadow)
    # attack_eval(attack_model, test_loader, args.device, method=args.attack_method_attention + "_quantized")
     
    mia_attention_threshold(args, config, path, q_target_model, dataloaders_target, q_shadow_model, dataloaders_shadow, q="quantized")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('RepQ-ViT', parents=[get_args_parser()])
    args = parser.parse_args()
    main()