# 锥筒颜色分类

基于 PyTorch 实现的三分类（蓝、红、黄）锥筒图像识别模型。网络采用多分支混合卷积结构（mixed_net），结合不同尺寸卷积核提取颜色与形状特征。

## 环境依赖

- Python 3.8+
- PyTorch (>=1.8)
- torchvision
- torchinfo (可选，仅用于查看模型结构)

## 数据准备

将数据集按以下结构放置在项目根目录：

dataset/
├── train/
│   ├── blue/      # 蓝色锥筒训练图片
│   ├── red/       # 红色锥筒训练图片
│   └── yellow/    # 黄色锥筒训练图片
├── test1/
│   ├── blue/
│   ├── red/
│   └── yellow/
└── test2/
    ├── blue/
    ├── red/
    └── yellow/

图片会自动缩放至 64×64，并归一化到 [-1, 1]。

## 模型结构 (mixed_net)

- **分支1**：两层 3×3 卷积 + 最大池化，输出 64 通道 16×16 特征图
- **分支2**：两层 5×5 卷积 + 最大池化，输出 64 通道 16×16 特征图
- **分支3**：平均池化 + 1×1 卷积 + 3×3 卷积 + 池化，输出 32 通道 16×16 特征图
- 拼接三个分支后，通过两个全连接层（160×16×16 → 256 → 3）输出类别 logits

## 训练

运行训练脚本（例如 `python train.py`）：

- 优化器：SGD（lr=0.01, momentum=0.9）
- 损失函数：交叉熵损失
- Epoch > 50 后，每 10 个 epoch 在 test1 上评估并保存最佳模型（基于 test1 准确率）
- 最佳模型保存在 `pth/` 目录，命名格式 `model_best_XX.XX.pth`

训练时可自动选择设备（CUDA > MPS > CPU）。

## 测试

使用 `test_model` 函数加载保存的模型进行评估：

```python
from my_net import mixed_net
model_path = "pth/model_best_94.47.pth"
test_model(model_path, test_loader1)
test_model(model_path, test_loader2)
```

输出总体准确率和每个类别的准确率。

## 注意事项

- 自定义网络模块文件建议命名为 `my_net.py`，避免与 PyPI 上的 `net` 包冲突。
- 测试时 `test_loader` 建议设置 `shuffle=False` 以保证评估可重复。
