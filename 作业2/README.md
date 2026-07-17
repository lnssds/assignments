# 作业二：DIP with PyTorch

杨洁 SC25005001

数字图像处理课程作业二，包含两部分内容：一是基于 PyTorch 实现泊松图像编辑（Poisson Image Editing），利用梯度域优化实现无缝融合；二是实现 Pix2Pix 图像翻译模型，使用全卷积网络（FCN）在 Facades 数据集上学习语义标签到建筑图像的映射。

## 1. 泊松图像编辑

### 环境依赖

```setup
pip install torch numpy opencv-python gradio
```

### 算法原理

泊松图像编辑是一种经典的梯度域图像融合方法，由 Pérez 等人于 2003 年提出。其核心思想是：不直接复制前景的像素值，而是在边界约束下让融合区域内的梯度尽可能与前景一致，从而实现无缝融合。

数学上，这等价于求解一个带狄利克雷边界条件的泊松方程：
```
Δf = div(g)    在融合区域 Ω 内
f  = f*        在区域边界 ∂Ω 上
```
其中 f 为待求的融合图像，g 为前景的梯度场，f* 为背景图像在边界上的像素值。离散化后这是一个大型线性方程组，本实现使用梯度下降法迭代求解。

### 实现细节

实现了两个核心函数：

**1. 多边形转掩膜（create_mask_from_points）**

将用户在 Gradio 界面上点击绘制的多边形顶点坐标转换为与图像等大的二值掩膜。使用 OpenCV 的 `cv2.fillPoly` 函数对多边形内部进行填充，多边形内像素值为 255，外部为 0。掩膜用于标记哪些像素需要参与优化，以及确定边界条件。

**2. 拉普拉斯损失（cal_laplacian_loss）**

使用 2D 卷积计算拉普拉斯算子，卷积核为：
```
[[0, 1, 0],
 [1,-4, 1],
 [0, 1, 0]]
```
损失函数定义为融合图像与前景图像在掩膜区域内的拉普拉斯差异的 L2 范数：
```
loss = || Laplacian(blended) - Laplacian(foreground) ||²
```
计算时仅对掩膜内的像素求损失，掩膜外的像素固定为背景值不参与梯度更新。

**3. 优化策略**

- 优化器：Adam，初始学习率 0.01
- 迭代次数：5000 步
- 学习率调度：在第 3300 步（约 2/3 处）将学习率乘以 0.1，前期快速下降，后期精细收敛
- 边界处理：掩膜边界像素直接复制背景值，不参与优化，保证边界处与背景连续

### 运行方式

```
cd 02-data_possion
python run_blending_gradio.py
```

操作流程：
1. 上传前景图像，在需要融合的物体周围点击放置多边形顶点
2. 顶点数不少于 3 个时，点击 **Close Polygon** 闭合多边形
3. 上传背景图像
4. 通过水平/垂直偏移滑块调整前景在背景中的位置
5. 点击 **Blend Images** 开始优化，约 5000 次迭代后输出结果

### 结果展示

在蒙娜丽莎数据集上的测试效果：

![poisson blending](https://aka.doubaocdn.com/s/PiKL1wnF4t)

可以看到，融合区域与背景之间没有明显的拼接边界，整体色调自然过渡，相比直接复制粘贴效果提升显著。

## 2. Pix2Pix

### 环境依赖

```setup
pip install torch torchvision numpy opencv-python pillow
```

### 网络结构

采用编码器-解码器（Encoder-Decoder）结构的全卷积网络：

- **编码器**：由 5 层步长为 2 的卷积组成，每层包含 Conv2d → BatchNorm2d → ReLU，通道数从 3 逐步增加到 256，空间尺寸从 256×256 下采样到 8×8
- **解码器**：由 5 层转置卷积组成，每层包含 ConvTranspose2d → BatchNorm2d → ReLU，通道数从 256 逐步减少到 3，空间尺寸从 8×8 上采样回 256×256
- **输出层**：最后一层使用 Tanh 激活函数，将输出限制在 [-1, 1] 范围，与归一化后的真值图像对应

### 训练配置

- **数据集**：Facades 建筑立面数据集，包含 400 张训练图像和 100 张测试图像，每张图像由真实建筑照片和对应的语义分割标签图左右拼接而成
- **损失函数**：L1 逐像素损失（论文中使用 GAN Loss + L1 Loss，本作业简化为仅使用 L1 Loss）
- **优化器**：Adam，学习率 0.001，动量参数 betas=(0.5, 0.999)
- **学习率调度**：StepLR，每 200 个 epoch 将学习率乘以 0.2
- **批次大小**：100
- **训练轮数**：300
- **数据增强**：随机水平翻转

### 运行方式

下载数据集：
```
cd 02-Pix2pix
bash download_facades_dataset.sh
```

开始训练：
```
python train.py
```

训练过程中：
- 每 5 个 epoch 保存一次训练集和验证集的对比结果图（输入 / 真值 / 输出），分别存放在 `train_results/` 和 `val_results/` 目录
- 每个 epoch 打印训练损失和验证损失
- 每 50 个 epoch 保存一次模型检查点到 `checkpoints/` 目录

### 结果展示

训练损失从约 0.75 下降至约 0.05，验证损失从约 0.71 下降至约 0.37。

**训练结果对比（第 13 epoch vs 第 279 epoch）：**

![train results](https://aka.doubaocdn.com/s/0BTx1wnF4t)

训练初期（第 13 epoch）输出较为模糊，只能看到大致的色块分布；训练后期（第 279 epoch）窗户、门等建筑构件的形状逐渐清晰，但由于未使用 GAN Loss，细节上仍有一定模糊。

## 致谢

- [Poisson Image Editing](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf) — Pérez et al., SIGGRAPH 2003
- [Image-to-Image Translation with Conditional Adversarial Nets](https://phillipi.github.io/pix2pix/) — Isola et al., CVPR 2017
- [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) — Long et al., CVPR 2015
