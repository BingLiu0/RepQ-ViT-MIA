from torchvision import datasets

# 加载训练集和测试集
train_set = datasets.CIFAR10(root='./data', train=True, download=True)
test_set = datasets.CIFAR10(root='./data', train=False, download=True)

# 查看训练集和测试集的总大小
print(f"训练集大小: {len(train_set)}")
print(f"测试集大小: {len(test_set)}")

# 查看每个类别的样本数量（train 和 test）
train_class_count = [0] * 10
test_class_count = [0] * 10

for _, label in train_set:
    train_class_count[label] += 1

for _, label in test_set:
    test_class_count[label] += 1

print("训练集中各类别的样本数量:", train_class_count)
print("测试集中各类别的样本数量:", test_class_count)