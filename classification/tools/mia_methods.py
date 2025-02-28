import torch
import numpy as np
from sklearn.utils import shuffle
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from .trainer import train_model
from .build_model import AttackModel
from .dataloaders import *
from .vit_rollout import VITAttentionRollout
from sklearn.metrics import accuracy_score, precision_score, recall_score

def add_gaussian_noise_to_patches(image_tensor, patch_size=16, num_patches=10, mean=0, stddev=0.25, device="cuda"):
    batch_size, channels, height, width = image_tensor.size()
    unfolded = torch.nn.functional.unfold(image_tensor, patch_size, stride=patch_size)
    selected_patches = torch.randperm(unfolded.size(2))[:num_patches]
    noise = torch.randn(batch_size, unfolded.shape[1], num_patches) * stddev + mean
    noise = noise.to(device)
    unfolded = unfolded.to(device)
    unfolded[:, :, selected_patches] += noise
    unfolded = unfolded.clamp(0,1.5)
    image_tensor = torch.nn.functional.fold(unfolded, (height, width), patch_size, stride=patch_size)
    return image_tensor


def switch_fused_attn(model):
    for name, block in model.blocks.named_children():
        block.attn.fused_attn = False
    return model


def init_config_model_attn(args, target_model, shadow_model):
    # if "attn" in args.atk_method or "roll" in args.atk_method:
    target_model = switch_fused_attn(target_model)
    shadow_model = switch_fused_attn(shadow_model)
    # target_model, shadow_model = target_model.to(args.device), shadow_model.to(args.device)
    target_rollout = VITAttentionRollout(target_model, head_fusion=args.head_fusion, discard_ratio=args.discard_ratio, device=args.device)
    shadow_rollout = VITAttentionRollout(shadow_model, head_fusion=args.head_fusion, discard_ratio=args.discard_ratio, device=args.device)
    return target_rollout, shadow_rollout


def get_attn(dataloaders, attention_rollout, noise=False, device = "cuda", out_atk=""):
    attn_metric = []
    for batch_idx, (inputs, labels) in enumerate(dataloaders):
        inputs = inputs.to(device)
        if noise:
            inputs = add_gaussian_noise_to_patches(inputs, device=device)
        mask = attention_rollout(inputs, out_atk)
        attn_metric.append(mask)
    attn = torch.cat(attn_metric, dim=0)
    return attn


def pearson_correlation(x, y, dim):
    mean_x = torch.mean(x, dim=dim)
    mean_y = torch.mean(y, dim=dim)
    cov = torch.sum((x - mean_x.unsqueeze(dim)) * (y - mean_y.unsqueeze(dim)), dim=dim) / x.size(dim)
    std_x = torch.std(x, dim=dim)
    std_y = torch.std(y, dim=dim)
    correlation = cov / (std_x * std_y)
    return correlation


def CrossEntropy(x, target):
    loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
    return loss


def get_data(args, attention_rollout, loader):
    attn_origin_train = get_attn(loader["train"], attention_rollout, noise=False, device =args.device, out_atk=args.attack_method_attention)
    attn_origin_test = get_attn(loader["test"], attention_rollout, noise=False, device =args.device, out_atk=args.attack_method_attention)
    # train_res = torch.zeros(attn_origin_train.shape[0]).to(args.device)
    # val_res = torch.zeros(attn_origin_val.shape[0]).to(args.device)
    train_res_list = []
    test_res_list = []
    for i in range(args.noise_repeat):
        attn_train= get_attn(loader["train"], attention_rollout, noise=True, device =args.device, out_atk=args.attack_method_attention)
        attn_test = get_attn(loader["test"], attention_rollout, noise=True, device =args.device, out_atk=args.attack_method_attention)
        # 皮尔逊相关系数
        if args.metric == "pearson":
            metric_train_repeat = pearson_correlation(attn_origin_train, attn_train, -1)
            metric_test_repeat = pearson_correlation(attn_origin_test, attn_test, -1)
        # 计算欧式距离
        elif args.metric == "Euclid":
            metric_train_repeat = torch.norm(attn_origin_train - attn_train, dim=1)
            metric_test_repeat = torch.norm(attn_origin_test - attn_test, dim=1)
            # metric_train_repeat = map_to_range(metric_train_repeat)
            # metric_val_repeat = map_to_range(metric_val_repeat)
        # 计算交叉熵
        elif args.metric == "CE":
            metric_train_repeat = CrossEntropy(attn_origin_train, attn_train)
            metric_test_repeat = CrossEntropy(attn_origin_test, attn_test)
            # metric_train_repeat = map_to_range(metric_train_repeat)
            # metric_val_repeat = map_to_range(metric_val_repeat)
        # 余弦相似度
        elif args.metric == "cos-sim":
            metric_train_repeat = torch.cosine_similarity(attn_origin_train,attn_train,dim=1)
            metric_test_repeat = torch.cosine_similarity(attn_origin_test, attn_test, dim=1)

        else:
            print("not support metric!")
            exit(0)
        train_res_list.append(metric_train_repeat)
        test_res_list.append(metric_test_repeat)

    train_res_threshold = sum(train_res_list) / args.noise_repeat
    test_res_threshold = sum(test_res_list) / args.noise_repeat
    threshold_data = torch.cat([train_res_threshold, test_res_threshold])
    threshold_label = torch.cat([torch.ones(train_res_threshold.shape[0], dtype=torch.long), torch.zeros(test_res_threshold.shape[0], dtype=torch.long)]).to(args.device)
    
    train_res_classifier = torch.stack(train_res_list, dim=-1)
    test_res_classifier = torch.stack(test_res_list, dim=-1)
    classifier_data = torch.cat([train_res_classifier, test_res_classifier])
    classifier_label = torch.cat([torch.ones(train_res_classifier.shape[0], dtype=torch.long), torch.zeros(test_res_classifier.shape[0], dtype=torch.long)]).to(args.device)

    return threshold_data, threshold_label, classifier_data, classifier_label
            
            
def find_best_threshold(data, labels):
    sorted_data, indices = torch.sort(data)
    sorted_labels = labels[indices]
    best_accuracy = 0.0
    best_threshold = 0.0
    for i in range(len(sorted_data) - 1):
        if sorted_labels[i+1] != 0:
            continue
        threshold = (sorted_data[i] + sorted_data[i+1]) / 2.0
        predicted_labels = (data <= threshold).to(torch.int32)
        # print(predicted_labels.sum().item())
        accuracy = (predicted_labels == labels).to(torch.float32).mean()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    # predicted_labels = (data >= best_threshold).to(torch.int32)
    # print(predicted_labels.sum().item())
    return best_threshold


def calculate_accuracy(data, labels, threshold):
    predicted_labels = (data <= threshold).to(torch.int32)
    
    labels = labels.cpu().numpy()
    predicted_labels = predicted_labels.cpu().numpy()
    
    precision = precision_score(y_true=labels, y_pred=predicted_labels, average = "macro", zero_division=1)
    recall = recall_score(y_true=labels, y_pred=predicted_labels, average = "macro", zero_division=1)
    accuracy = accuracy_score(y_true=labels, y_pred=predicted_labels)
    
    return accuracy, precision, recall
            
            
def mia_attention_threshold(args, config, path, target_model, dataloaders_target, shadow_model, dataloaders_shadow, q=""):
    target_rollout, shadow_rollout = init_config_model_attn(args, target_model, shadow_model)
   
    data_train_attack, label_train_attack, _, _ = get_data(args, shadow_rollout, dataloaders_shadow)
    
    threshold = find_best_threshold(data_train_attack, label_train_attack)
  
    data_test_attack, label_test_attack, _, _ = get_data(args, target_rollout, dataloaders_target)
    
    accuracy, precision, recall = calculate_accuracy(data_test_attack, label_test_attack, threshold)
    
    np.save(path + f"/res_precision_{args.attack_method_attention}_{q}_threshold.npy", precision)
    np.save(path + f"/res_recall_{args.attack_method_attention}_{q}_threshold.npy", recall)
    np.save(path + f"/res_accuracy_{args.attack_method_attention}_{q}_threshold.npy", accuracy)
    
    data_test_attack = data_test_attack.cpu().numpy()
    label_test_attack = label_test_attack.cpu().numpy()
    np.save(path + f"/res_label_attack_{args.attack_method_attention}_{q}_threshold.npy", label_test_attack)
    np.save(path + f"/res_soft_pred_attack_{args.attack_method_attention}_{q}_threshold.npy", data_test_attack)
    
    print(f"END {args.attack_method_attention} THREASHOLD")
    

def mia_attention_classifier(args, config, target_model, dataloaders_target, shadow_model, dataloaders_shadow):
    config.attack_learning.epochs = 25
    config.attack_learning.learning_rate = 0.0005
    
    if args.metric == "Euclid" and "last_attention" in args.attack_method_attention:
        config.attack_learning.epochs = 200
        config.attack_learning.learning_rate = 0.00025
        
    elif args.metric == "CE" and "last_attention" in args.attack_method_attention:
        config.attack_learning.epochs = 50
        config.attack_learning.learning_rate = 0.0005
        
    target_rollout, shadow_rollout = init_config_model_attn(args, target_model, shadow_model)
 
    _, _, data_train_attack, label_train_attack = get_data(args, shadow_rollout, dataloaders_shadow)
    train_dataset = torch.utils.data.TensorDataset(data_train_attack, label_train_attack)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.attack_learning.train_batch_size, shuffle=True)

    _, _, data_test_attack, label_test_attack = get_data(args, target_rollout, dataloaders_target)
    test_dataset = torch.utils.data.TensorDataset(data_test_attack, label_test_attack)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.attack_learning.train_batch_size, shuffle=False)

    dataloaders_attack = {"train": train_loader, "test": test_loader}
    dataset_sizes_attack = {"train": len(train_dataset), "test": len(test_dataset)}
    
    print("Training mia_rollout attack model ..........................")
    attack_model = AttackModel(args.noise_repeat).to(args.device)
    optimizer = optim.SGD(attack_model.parameters(), lr=config.attack_learning.learning_rate, momentum=config.attack_learning.momentum, weight_decay=config.attack_learning.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config.attack_learning.decrease_lr_factor, gamma=config.attack_learning.decrease_lr_every)
    criterion = nn.CrossEntropyLoss()
    attack_model, epoch_loss_acc_attack = train_model(attack_model, criterion, optimizer, exp_lr_scheduler, dataloaders_attack, dataset_sizes_attack, num_epochs=config.attack_learning.epochs)
    
    print(f"END {args.attack_method_attention} CLASSIFIER")
    
    return attack_model, test_loader


def mia_base(args, config, target_model, dataloaders_target, shadow_model, dataloaders_shadow):
    data_test_set, label_test_set = get_data_for_mia_base(target_model, dataloaders_target, args.device)
    data_train_set, label_train_set = get_data_for_mia_base(shadow_model, dataloaders_shadow, args.device)
    data_train_set, label_train_set = shuffle(data_train_set, label_train_set, random_state=args.seed)
    data_test_set, label_test_set = shuffle(data_test_set, label_test_set, random_state=args.seed)
    
    data_train_set_tensor = torch.from_numpy(data_train_set)
    label_train_set_tensor = torch.from_numpy(label_train_set)
    
    data_test_set_tensor = torch.from_numpy(data_test_set)
    label_test_set_tensor = torch.from_numpy(label_test_set)
    
    data_train_attack = data_train_set_tensor
    label_train_attack = label_train_set_tensor
    dataset_train_attack = torch.utils.data.TensorDataset(data_train_attack, label_train_attack)
    train_loader_attack = torch.utils.data.DataLoader(dataset_train_attack, batch_size=config.attack_learning.train_batch_size, shuffle=True)
    
    data_test_attack = data_test_set_tensor
    label_test_attack = label_test_set_tensor
    dataset_test_attack = torch.utils.data.TensorDataset(data_test_attack, label_test_attack)
    test_loader_attack = torch.utils.data.DataLoader(dataset_test_attack, batch_size=config.attack_learning.test_batch_size, shuffle=False)

    dataloaders_attack = {"train": train_loader_attack, "test": test_loader_attack}
    dataset_sizes_attack = {"train": len(dataset_train_attack), "test": len(dataset_test_attack)}

    print("Training mia_base attack model")
    attack_model = AttackModel(config.general.num_classes).to(args.device)
    optimizer = optim.SGD(attack_model.parameters(), lr=config.attack_learning.learning_rate, momentum=config.attack_learning.momentum, weight_decay=config.attack_learning.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config.attack_learning.decrease_lr_factor, gamma=config.attack_learning.decrease_lr_every)
    criterion = nn.CrossEntropyLoss()
    attack_model, epoch_loss_acc_attack = train_model(attack_model, criterion, optimizer, exp_lr_scheduler, dataloaders_attack, dataset_sizes_attack, num_epochs=config.attack_learning.epochs)
    
    print("END MIA_BASE")
    
    return attack_model, test_loader_attack
