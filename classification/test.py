from torchvision import datasets

# 加载训练集和测试集
# train_set = datasets.CIFAR10(root='./data', train=True, download=True)
# test_set = datasets.CIFAR10(root='./data', train=False, download=True)
train_set = datasets.ImageFolder("./data/ISIC2018/train")
test_set = datasets.ImageFolder("./data/ISIC2018/test")
val_set = datasets.ImageFolder("./data/ISIC2018/val")                                     
# 查看训练集和测试集的总大小
print(f"训练集大小: {len(train_set)}")
print(f"测试集大小: {len(test_set)}")
print(f"验证集大小: {len(val_set)}")
# 查看每个类别的样本数量（train 和 test）
train_class_count = [0] * 7
test_class_count = [0] * 7
val_class_count = [0] * 7

for _, label in train_set:
    train_class_count[label] += 1

for _, label in test_set:
    test_class_count[label] += 1

for _, label in val_set:
    val_class_count[label] += 1
    
print("训练集中各类别的样本数量:", train_class_count)
print("测试集中各类别的样本数量:", test_class_count)
print("验证中各类别的样本数量:", val_class_count)
# import os

# cpu_cores = os.cpu_count()
# print(f"CPU 核心数: {cpu_cores}")

# import timm
# import torch

# # 创建模型
# model = timm.create_model("vit_base_patch16_224", pretrained=True)

# # 输入张量，例如一个大小为 (batch_size, channels, height, width) 的图像
# input_tensor = torch.randn(10, 3, 224, 224)  # 假设输入是一个 batch_size=1 的 224x224 RGB 图像

# # 获取中间特征输出
# features = model.forward_features(input_tensor)

# print(features.shape) 