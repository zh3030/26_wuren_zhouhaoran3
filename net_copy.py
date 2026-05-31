import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary
import os

class mixed_net(nn.Module):
    def __init__(self):
        super(mixed_net, self).__init__()
        # 分支1：3x3 卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),       # 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)        # 32 -> 16
        )
        # 分支2：5x5 卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # 分支3：平均池化 + 1x1 卷积
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),  # 尺寸不变
            nn.Conv2d(3, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),       # 64 -> 32
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)        # 32 -> 16
        )
        # 全连接层（修正输入维度）
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(160 * 16 * 16, 256),   # 160通道 × 16 × 16
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        out1 = self.branch1(x)   # (batch, 64, 16, 16)
        out2 = self.branch2(x)   # (batch, 64, 16, 16)
        out3 = self.branch3(x)   # (batch, 32, 16, 16)
        out = torch.cat([out1, out2, out3], dim=1)  # (batch, 160, 16, 16)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    # 图像转换
    transforms = transforms.Compose(
        [
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    
    # 超参数设置
    BATCH_SIZE = 1024
    EPOCH = 100

    # 加载数据
    trainset = datasets.ImageFolder(root=r'dataset/train', transform=transforms)
    testset1 = datasets.ImageFolder(root=r'dataset/test1', transform=transforms)
    testset2 = datasets.ImageFolder(root=r'dataset/test2', transform=transforms)

    print(f"训练集图片数量: {len(trainset)}")
    print(f"测试集1图片数量: {len(testset1)}")
    print(f"测试集2图片数量: {len(testset2)}")
    
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader1 = DataLoader(testset1, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader2 = DataLoader(testset2, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # 自动选择设备（优先CUDA，其次MPS，最后CPU）
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    net = mixed_net().to(device)
    
    # 打印网络信息
    summary(net, input_size=(1, 3, 64, 64), device=device)
    print(f'标签对应的ID: {trainset.class_to_idx}')

    # 设置优化器、损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # 开始训练
    print("Start Training")
    max_correct = 0.0  # 记录最佳测试集1准确率
    for epoch in range(EPOCH):
        print(f"Epoch {epoch + 1}/{EPOCH}")
        net.train()
        train_loss = 0.0
        total_samples = 0
        
        for batch_id, (datas, labels) in enumerate(train_loader):
            datas, labels = datas.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(datas)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 累积总损失（用于计算平均损失）
            train_loss += loss.item() * datas.size(0)
            total_samples += datas.size(0)
        
        avg_loss = train_loss / total_samples  # 当前epoch平均损失
        
        # 定期评估和保存
        if epoch > 5 and (epoch + 1) % 10 == 0:
            os.makedirs("pth", exist_ok=True)
            temp_path = "pth/modeltemp.pth"
            torch.save(net.state_dict(), temp_path)
            
            # 加载临时模型进行评估
            model = mixed_net().to(device)
            model.load_state_dict(torch.load(temp_path, map_location=device))
            model.eval()
            
            correct1 = 0
            correct2 = 0
            total1 = 0
            total2 = 0
            
            with torch.no_grad():
                for datas1, labels1 in test_loader1:
                    datas1, labels1 = datas1.to(device), labels1.to(device)
                    output_test1 = model(datas1)
                    _, predicted1 = torch.max(output_test1.data, dim=1)
                    total1 += predicted1.size(0)
                    correct1 += (predicted1 == labels1).sum().item()
                
                for datas2, labels2 in test_loader2:
                    datas2, labels2 = datas2.to(device), labels2.to(device)
                    output_test2 = model(datas2)
                    _, predicted2 = torch.max(output_test2.data, dim=1)
                    total2 += predicted2.size(0)
                    correct2 += (predicted2 == labels2).sum().item()
            
            c1 = 100.0 * correct1 / total1
            c2 = 100.0 * correct2 / total2
            print(f"epoch:{epoch + 1}\tavg_loss:{avg_loss:.5f}\t"
                  f"test1_acc:{c1:.2f}%\ttest2_acc:{c2:.2f}%")
            
            # 保存最佳模型（基于测试集1准确率）
            if c1 > max_correct:
                max_correct = c1
                best_path = f"pth/model_best_{max_correct:.2f}.pth"
                torch.save(net.state_dict(), best_path)
                print(f"Save best model to {best_path}")