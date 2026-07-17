# 图像几何变换实现

数字图像处理课程作业一，实现了基础图像几何变换和基于移动最小二乘（MLS）的点引导图像变形。

![teaser](https://raw.githubusercontent.com/YudongGuo/DIP-Teaching/d136274e3262b50392808694d99169cf6f8e6e76/Assignments/01_ImageWarping/pics/teaser.png)

## 环境依赖

```setup
pip install opencv-python numpy gradio
```

## 运行方式

### 1. 基础几何变换

```
python 作业1.1.py
```

启动 Gradio 交互界面，支持实时调整：
- **Scale**：缩放比例，0.1× ~ 2.0×
- **Rotation**：旋转角度，-180° ~ 180°
- **Translation X/Y**：平移量，-300 ~ 300 像素
- **Horizontal Flip**：水平翻转

所有变换均以图像中心为基准进行。仿射矩阵复合顺序为：平移到原点 → 缩放 → 旋转 → 平移回中心并叠加偏移 → （可选）水平翻转。越界像素用白色填充。

**测试图片：**

![test image](https://aka.doubaocdn.com/s/V1kA1wnEqn)

**效果展示：**

![basic transform](https://raw.githubusercontent.com/YudongGuo/DIP-Teaching/d136274e3262b50392808694d99169cf6f8e6e76/Assignments/01_ImageWarping/pics/global_demo.gif)

### 2. 点引导图像变形（MLS）

```
python 作业1.2.py
```

使用方法：
1. 上传图像
2. 在图像上交替点击放置源点（蓝色）和目标点（红色），绿色箭头表示位移方向
3. 点击 **Run Warping** 执行 MLS 变形
4. 点击 **Clear Points** 重置

实现遵循 MLS 论文：对每个像素通过加权最小二乘计算局部仿射变换，权重采用反距离平方 `1/dist²`。最终使用 `cv2.remap` 双线性插值完成重映射。

**效果展示：**

![point deformation](https://raw.githubusercontent.com/YudongGuo/DIP-Teaching/d136274e3262b50392808694d99169cf6f8e6e76/Assignments/01_ImageWarping/pics/point_demo.gif)

## 致谢

- [Image Deformation Using Moving Least Squares](https://people.engr.tamu.edu/schaefer/research/mls.pdf)
