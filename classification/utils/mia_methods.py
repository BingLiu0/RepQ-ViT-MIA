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


def add_gaussian_noise_to_patches(image_tensor, patch_size=16, num_patches=10, mean=0, stddev=0.25):
    batch_size, channels, height, width = image_tensor.size()
    unfolded = torch.nn.functional.unfold(image_tensor, patch_size, stride=patch_size)
    selected_patches = torch.randperm(unfolded.size(2))[:num_patches]
    noise = torch.randn(batch_size, unfolded.shape[1], num_patches) * stddev + mean
    noise = noise.to(args.device)
    unfolded = unfolded.to(args.device)
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


def get_attn(model, dataloaders, attention_rollout, noise=False, out_atk="out"):
    attn_metric = []
    for batch_idx, (inputs, labels) in enumerate(dataloaders):
        inputs = inputs.to(args.device)
        if noise:
            inputs = add_gaussian_noise_to_patches(inputs)
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


def get_data(model, attention_rollout, loader, atk_method):
    attn_origin_train = get_attn(model, loader["train"], attention_rollout, noise=False, out_atk=atk_method)
    attn_origin_test = get_attn(model, loader["test"], attention_rollout, noise=False, out_atk=atk_method)
    # train_res = torch.zeros(attn_origin_train.shape[0]).to(args.device)
    # val_res = torch.zeros(attn_origin_val.shape[0]).to(args.device)
    train_res_list = []
    test_res_list = []
    for i in range(args.noise_repeat):
        attn_train= get_attn(model, loader["train"], attention_rollout, noise=True, out_atk=atk_method)
        attn_test = get_attn(model, loader["test"], attention_rollout, noise=True, out_atk=atk_method)
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


    if "nn" not in atk_method:
        train_res = sum(train_res_list) / args.noise_repeat
        test_res = sum(test_res_list) / args.noise_repeat
    else:
        train_res = torch.stack(train_res_list, dim=-1)
        test_res = torch.stack(test_res_list, dim=-1)

    if "metric" in atk_method:
        return train_res, test_res

    data = torch.cat([train_res, test_res])
    label = torch.cat([torch.ones(train_res.shape[0], dtype=torch.long), torch.zeros(test_res.shape[0], dtype=torch.long)]).to(args.device)
    # dataset = TensorDataset(data, target)
    # return dataset
    return data, label


# def get_loader(dataset, config, is_target, cif=True, set = 0):
#     global image_size
#     image_size = 224
#     train_set = test_set = None
#     if 'cifar' in dataset:
#         train_set = VITdataset(root_dir=config.path.data_path,
#                                split='train',
#                                num_class=config.patch.num_classes,
#                                nums_per_class=config.patch.nums_per_class,
#                                random_seed=config.general.seed,
#                                is_target=is_target)
#         test_set = VITdataset(root_dir=config.path.data_path,
#                              split='test',
#                              num_class=config.patch.num_classes,
#                              nums_per_class=config.patch.nums_per_class,
#                              random_seed=config.general.seed,
#                              is_target=is_target)
#         # train_set = Subset(train_set, range(100))
#         # val_set = Subset(val_set, range(100))
#     elif dataset == 'ImageNet':
#         train_set = imageNet(root_dir=config.path.data_path, split='train',
#                              num_class=config.patch.num_classes,
#                              nums_per_class=config.patch.nums_per_class,
#                              is_target=is_target,
#                              random_seed=config.general.seed,
#                              set=set)
#         test_set = imageNet(root_dir=config.path.data_path, split='test',
#                            num_class=config.patch.num_classes,
#                            nums_per_class=config.patch.nums_per_class,
#                            is_target=is_target,
#                            random_seed=config.general.seed,
#                            set=set)
#     elif dataset == 'ImageNet10':
#         train_set = imageNet10(root_dir=config.path.data_path, split='train', is_target=is_target,
#                                seed=config.general.seed)
#         test_set = imageNet10(root_dir=config.path.data_path, split='test', is_target=is_target, seed=config.general.seed)
#     elif dataset == 'ImageNet100':
#         train_set = imageNet100(root_dir=config.path.data_path, split='train',
#                              num_class=config.patch.num_classes,
#                              nums_per_class=config.patch.nums_per_class,
#                              is_target=is_target,
#                              random_seed=config.general.seed,
#                              set=set)
#         test_set = imageNet100(root_dir=config.path.data_path, split='test',
#                            num_class=config.patch.num_classes,
#                            nums_per_class=config.patch.nums_per_class,
#                            is_target=is_target,
#                            random_seed=config.general.seed,
#                            set=set)
#         # train_set = Subset(train_set, range(1000))
#         # val_set = Subset(val_set, range(1000))
#     elif dataset == 'cinic10':
#         train_set = cinic(root_dir=config.path.data_path, split='train', is_target=is_target)
#         test_set = cinic(root_dir=config.path.data_path, split='test', is_target=is_target)
#     elif dataset == 'ISIC2018':
#         # print(is_target)
#         train_set = ISIC2018(root_dir=config.path.data_path, split='train')
#         test_set = ISIC2018(root_dir=config.path.data_path, split='test') + ISIC2018(root_dir=config.path.data_path, split='test')
#         class_indices = {class_idx: [] for class_idx in range(len(train_set.dataset.classes))}
#         for idx, (_, class_idx) in enumerate(train_set.dataset.imgs):
#             class_indices[class_idx].append(idx)
#         subsets_indices_1 = []
#         subsets_indices_2 = []
#         for class_idx, indices in class_indices.items():
#             indices_1, indices_2 = train_test_split(indices, test_size=0.5, random_state=1001)
#             subsets_indices_1.extend(indices_1)
#             subsets_indices_2.extend(indices_2)
#         subset1 = Subset(train_set, subsets_indices_1)
#         subset2 = Subset(train_set, subsets_indices_2)
#         if is_target:
#             train_set = subset1 + test_set
#             test_set = subset2
#         else:
#             train_set = subset2 + test_set
#             test_set = subset1     
            
            
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

            
def mia_rollout_threshold(args, config, target_model, dataloaders_target, shadow_model, dataloaders_shadow):
    target_rollout, shadow_rollout = init_config_model_attn(args, target_model, shadow_model)
    # sha_loader, sha_size = get_loader(args.dataset, config, is_target=False)
    data_train_attack, label_train_attack = get_data(shadow_model, shadow_rollout, dataloaders_shadow, args.atk_method)
    thr = find_best_threshold(data_train_attack, label_train_attack)
    # tar_loader, tar_size = get_loader(args.dataset, config, is_target=True)
    data_test_attack, label_test_attack = get_data(target_model, target_rollout, dataloaders_target, args.atk_method)
    # np.save("results/{}_{}.npy".format(args.dataset, args.model), tar_dataset.cpu().numpy())
    test_acc, pre, rec = calculate_accuracy(data_test_attack, label_test_attack, thr)
    # val_acc, pre, rec = torch.tensor(0), torch.tensor(0), torch.tensor(0)
    print("model: {}".format(args.model))
    print("shadow: {}".format(args.shadow))
    pad = "Output"
    if args.atk_method == "roll":
        pad = "Attn rollout"
    elif args.atk_method == "last_attn":
        pad = "Last_attn"
    print("{} attack acc:{:.4f}\nprecision: {:.4f}\nRecall: {:.4f}".format(pad,test_acc, pre,rec))
    return test_acc.item(), pre.item(), rec.item()

    print("END MIA_ROLLOUT_THRESHOLD")
    

def mia_rollout_classifier(args, config, target_model, dataloaders_target, shadow_model, dataloaders_shadow):
    ###
    args.noise_repeat = 3
    args.epochs = 25
    args.lr = 0.0005
    ###
    if args.metric == "Euclid" and "attn" in args.atk_method:
        # args.noise_repeat = 3
        args.epochs = 200
        args.lr = 0.00025
    elif args.metric == "CE" and "attn" in args.atk_method:
        # args.noise_repeat = 3
        args.epochs = 50
        args.lr = 0.0005
    target_rollout, shadow_rollout = init_config_model_attn(args, target_model, shadow_model)
    # sha_loader, sha_size = get_loader(args.dataset, config, is_target=False)
    data_train_attack, label_train_attack = get_data(shadow_model, shadow_rollout, dataloaders_shadow, args.atk_method)
    train_dataset = torch.utils.data.TensorDataset(data_train_attack, label_train_attack)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    # # # # #
    # torch.save(train_loader, "train/{}{}.pt".format(args.model, args.shadow))
    # torch.save(tra_dataset, "pt/tra_dataset.pt")

    # train_loader = torch.load("train/{}{}.pt".format(args.model, args.shadow))
    # tra_dataset = torch.load("pt/tra_dataset.pt")

    atk_model = Classifier(args.noise_repeat)
    # atk_model.load_state_dict(torch.load("atk_model.pth"))
    # torch.save(atk_model.state_dict(), "atk_model.pth")
    atk_model = train(atk_model, train_loader, 30000, args.epochs)

    # val_loader = get_tensor_loader(args, config, istarget =True)

    # tar_loader, tar_size = get_loader(args.dataset, config, is_target=True)
    data_test_attack, label_test_attack = get_data(target_model, target_rollout, dataloaders_target, args.atk_method)
    test_dataset = torch.utils.data.TensorDataset(data_test_attack, label_test_attack)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
    # # # # #
    # torch.save(train_loader, "val/{}{}.pt".format(args.model, args.shadow))
    # torch.save(val_dataset, "pt/val_dataset.pt")

    # val_loader = torch.load("val/{}{}.pt".format(args.model, args.shadow))
    # val_dataset = torch.load("pt/val_dataset.pt")

    acc, pre, rec = predict(atk_model, test_loader, 30000)
    print("model: {}".format(args.model))
    print("shadow: {}".format(args.shadow))
    print(acc)
    pad = "Output"
    if args.atk_method == "roll":
        pad = "Attn rollout"
    elif args.atk_method == "last_attn":
        pad = "Last_attn"
    print("{} attack acc:{:.4f}\nprecision: {:.4f}\nRecall: {:.4f}".format(pad,acc, pre,rec))
    return acc, pre, rec

    print("END MIA_ROLLOUT_CLASSIFIER")

def mia_base(args, path, target_model, dataloaders_target, shadow_model, dataloaders_shadow):
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
    train_loader_attack = torch.utils.data.DataLoader(dataset_train_attack, batch_size=args.attack_batch_size, shuffle=True)
    
    data_test_attack = data_test_set_tensor
    label_test_attack = label_test_set_tensor
    dataset_test_attack = torch.utils.data.TensorDataset(data_test_attack, label_test_attack)
    test_loader_attack = torch.utils.data.DataLoader(dataset_test_attack, batch_size=args.attack_batch_size, shuffle=False)

    dataloaders_attack = {"train": train_loader_attack, "test": test_loader_attack}
    dataset_sizes_attack = {"train": len(dataset_train_attack), "test": len(dataset_test_attack)}

    print("Training mia_base attack model")
    model_attack = AttackModel(args.num_class).to(args.device)
    optimizer = optim.SGD(model_attack.parameters(), lr=args.attack_learning_rate, momentum=args.attack_momentum, weight_decay=args.attack_weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.attack_decrease_lr_factor, gamma=args.attack_decrease_lr_every)
    criterion = nn.CrossEntropyLoss()
    model_attack, epoch_loss_acc_attack = train_model(model_attack, criterion, optimizer, exp_lr_scheduler, dataloaders_attack, dataset_sizes_attack, num_epochs=args.attack_epochs)
    np.save(path + "/res_epoch_loss_acc_attack_mia_base"+".npy", epoch_loss_acc_attack)
    
    return model_attack, test_loader_attack

    print("END MIA_BASE")
