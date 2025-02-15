# ResShuffle超分辨率模型简介

本项目实现了一个基于 PyTorch 的超分辨率模型，旨在将低分辨率图像恢复为高分辨率图像。该模型主要融合了 [EDSR](https://arxiv.org/abs/1707.02921) 中的残差块设计和 [ESPCN](https://arxiv.org/abs/1609.05158) 中的 PixelShuffle 上采样策略，从而实现高效且准确的超分辨率重建。

---

## 模型架构

整个模型由以下几个主要部分构成：

### 1. 特征提取
- **输入卷积层**  
  使用一个大尺寸卷积核（kernel size = 9）来提取图像的初步特征，并通过 ReLU 激活函数进行非线性变换。

### 2. 多个残差块 (Residual Blocks)
- **残差学习**  
  模型中采用多个残差块，每个残差块包含两个卷积层和一个中间的 ReLU 激活函数。  
- **跳跃连接**  
  每个残差块在前向传播时，将输入直接与经过卷积变换后的特征相加，缓解梯度消失问题并加速模型收敛。

### 3. 全局跳跃连接与中间卷积
- **中间卷积层**  
  经过残差块后，通过一个卷积层处理特征图。  
- **全局跳跃连接**  
  将中间卷积层的输出与初始提取的特征相加，有助于保留原始图像的低级细节信息。

### 4. 上采样模块 (Upsample Blocks)
- **PixelShuffle 策略**  
  上采样模块先通过卷积层将特征图的通道数扩展，再利用 PixelShuffle 操作将通道信息重排列为空间信息，从而实现图像尺寸的放大。  
- **灵活的放大倍数**  
  模型支持不同的上采样因子（如 2 倍或 4 倍），当采用 4 倍上采样时，通常使用两个连续的 2 倍上采样块。

### 5. 重构层
- **输出层**  
  最后一个卷积层（同样采用 kernel size = 9）将上采样后的特征图映射回原始图像的通道数，生成高分辨率图像。

---
## 训练设置
在训练过程中，我们针对超分辨率任务的特点和模型结构进行了精心的参数设计。下面对各主要参数的选择理由做详细说明：

- **num_epochs (默认400)**  
  为了确保模型有足够的时间进行学习和优化，我们设置了较高的epoch数量（400个epoch）。同时结合早停机制，当连续一定epoch内没有性能提升时提前终止训练，以防止过拟合并节省计算资源。

- **batch_size (默认32)**  
  批次大小设置为32，旨在平衡训练速度和内存消耗。32的batch size既能保证梯度计算的稳定性，又不会过多占用GPU显存。

- **lr (学习率，默认2e-4)**  
  初始学习率选择2e-4，这是一个在超分辨率任务中经多次实验验证有效的值，既能使模型稳定收敛，又可以避免在训练初期出现剧烈震荡。

- **upscale_factor (默认4)**  
  设置上采样因子为4，表示模型将低分辨率图像放大4倍，这一设定符合常见的超分辨率需求，能够更好地恢复图像细节。

- **num_features (默认64)**  
  在输入层后采用64个特征图，这个设置有助于捕捉图像的丰富细节信息，同时又不会引入过多计算开销。

- **num_residuals (默认16)**  
  模型中采用16个残差块，以加深网络结构，提高超分辨率重建的能力。较深的残差结构能够有效缓解梯度消失问题，并提高模型的细节恢复效果。

- **num_workers (默认4)**  
  数据加载过程中使用4个子进程来加速数据读取和预处理，确保在训练时不会因数据读取而成为瓶颈。

- **step_size (默认50) 和 gamma (默认0.5)**  
  使用学习率调度器，每50个epoch将学习率乘以0.5，使学习率逐步下降。这种策略有助于在训练后期细化模型参数，稳定收敛，防止在模型收敛后出现震荡现象。

## 效果评估

使用DIV2K_train_HR中的800张2K分辨率进行训练，使用DIV2K_vaild_HR进行评估，此验证集有100张2k图像
先使用bicubic下采样得到1/4的分辨率，然后使用模型进行四倍放大分辨率，恢复原分辨率。这种情况下，平均PSNR: 27.5790，平均SSIM: 0.8031

下面展示了几组超分辨率重建的对比效果（建议点击放大观看）：

| 原始图像 | 低分辨率输入 | 重建结果 |
|:---:|:---:|:---:|
| ![原图](/show/0803.png) | ![低分辨率](/show/0803_lr.png) | ![重建结果](/show/0803_SR.png) |
| ![原图](/show/0815.png) | ![低分辨率](/show/0815_lr.png) | ![重建结果](/show/0815_SR.png) |
| ![原图](/show/0824.png) | ![低分辨率](/show/0824_lr.png) | ![重建结果](/show/0824_SR.png) |
| ![原图](/show/0894.png) | ![低分辨率](/show/0894_lr.png) | ![重建结果](/show/0894_SR.png) |

## 环境设置

PyTorch版本: 2.1.1
CUDA版本: 12.1
Pillow版本: 10.0.1
torchvision版本: 0.16.1
OpenCV版本: 4.11.0
NumPy版本: 1.26.3
scikit-image版本: 0.24.0
Python版本: 3.9.18

## 模型亮点

- **深度残差学习**  
  通过多个残差块和全局跳跃连接，有效提高了模型对图像细节的恢复能力，缓解了深层网络训练中的梯度消失问题。

- **高效上采样**  
  使用 PixelShuffle 实现上采样，不仅可以减少计算量，还能更好地重构高频细节，提升图像质量。

- **模块化设计**  
  模型结构清晰，各模块（特征提取、残差块、上采样、重构层）独立且易于扩展，为后续的模型改进和应用提供了良好的基础。

---

