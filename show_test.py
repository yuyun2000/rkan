import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 假设你有一个一维数组，长度为 784
array_1d = np.load('./testin.npy')

# 将一维数组重塑为 28x28 的二维数组
array_2d = array_1d.reshape((28, 28))

# 将 NumPy 数组转换为 PIL 图像
image = Image.fromarray((array_2d * 255).astype(np.uint8))

# 展示图像
image.show()

# 如果你想使用 matplotlib 展示图像
plt.imshow(array_2d, cmap='gray')  # 使用灰度色图
plt.axis('off')  # 关闭坐标轴
plt.show()