import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import models

#device=torch.device("cuda")

torch.cuda.empty_cache()

# 定义交叉验证参数
k = 5
epochs = 100
batch_size =64
lr = 1e-3
lr_fine=1e-4
steps=25
gamma=0.1

# 文件路径
data_dir = "HuSHem"

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
# 创建 ImageFolder 数据集实例
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
net = models.vgg16()

net.classifier[0]=nn.Flatten()
net.classifier[1]=nn.Sequential(
    nn.Linear(25088,128),
    nn.ReLU(inplace=True)
)
net.classifier[2]=nn.Dropout(0.4)
net.classifier[3]=nn.Sequential(
    nn.Linear(128,128),
    nn.ReLU(inplace=True)
)
net.classifier[4]=nn.Dropout(0.4)
net.classifier[5]=nn.Sequential(
    nn.Linear(128,4),
    nn.Softmax(dim=1)
)
del net.classifier[6]
dict=torch.load("demo_method/vGG16_method83.79629629629629.pth")
net.load_state_dict(dict)
# 先冻结所有层
for param in net.parameters():
    param.requires_grad = False

# 然后解冻第17层以及之后的所有层
for i, param in enumerate(net.features.parameters()):
    if i >= 17:
        param.requires_grad = True

# 解冻分类器部分
for param in net.classifier.parameters():
    param.requires_grad = True

net=net.cuda()
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
criterion=criterion.cuda()
optimizer = optim.SGD(net.classifier.parameters(), lr=lr, momentum=0.9)
optimizer_fine = optim.SGD(net.parameters(), lr=lr_fine, momentum=0.9)
scheduler_train = StepLR(optimizer, step_size=steps, gamma=gamma)
scheduler_fine_tuning = StepLR(optimizer_fine, step_size=steps, gamma=gamma)

# 初始化 k-fold
kf = KFold(n_splits=k, shuffle=True)

# 交叉验证训练
for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
    # 数据分为训练集和验证集
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    # 训练模型
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs=inputs.cuda()
            labels=labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler_train.step()
        net.eval()
    #微调阶段
    for epoch in range(epochs):
        running_loss = 0.0
        net.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs=inputs.cuda()
            labels=labels.cuda()
            optimizer_fine.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_fine.step()
            running_loss += loss.item()
        scheduler_fine_tuning.step()
        net.eval()
            # 输出训练结果
        print("Fold [%d]/[%d] Epoch [%d]/[%d] Loss: %.3f" % (fold + 1, k, epoch + 1, epochs, running_loss / (i + 1)))

        # 验证模型
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images=images.cuda()
                labels=labels.cuda()
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print('Validation Accuracy: %d %%' % (accuracy),end="------")

        #整体数据集上验证
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images=images.cuda()
                labels=labels.cuda()
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print('test Accuracy: %d %%' % (accuracy))
        if accuracy>85 :
            torch.save(net.state_dict(), "demo_method/vGG16_method{}.pth".format(accuracy))
