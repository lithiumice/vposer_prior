import glob
import numpy as np
import cv2
import math

# 获取指定目录下的所有图片文件
# image_files = glob.glob('debug/latent_vis_for_rebuttal/sample_vis_loco_grid/*.png')
image_files = glob.glob('debug/latent_vis_for_rebuttal/sample_vis_loco_grid_blue/*.png')

# 读取所有图片
images = [cv2.imread(file)[:,200:600,:] for file in image_files]

num_cols = 14
max_row = 10
images = images[:(num_cols*max_row)]

# 确保所有图片大小相同
assert all(img.shape == images[0].shape for img in images), "All images must have the same dimensions"

# 获取单个图片的尺寸
img_height, img_width, _ = images[0].shape

# 计算行数
num_images = len(images)
num_rows = math.ceil(num_images / num_cols)

# if num_rows>8: num_rows=8

# 创建空白画布
canvas = np.zeros((num_rows * img_height, num_cols * img_width, 3), dtype=np.uint8)

# 将图片填充到画布上
for i, img in enumerate(images):
    row = i // num_cols
    col = i % num_cols
    canvas[row*img_height:(row+1)*img_height, col*img_width:(col+1)*img_width] = img

# 保存结果
cv2.imwrite('debug/gallery_output.png', canvas)