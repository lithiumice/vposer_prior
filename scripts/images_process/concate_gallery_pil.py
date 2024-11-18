import glob
from PIL import Image
import math

# 获取指定目录下的所有图片文件
image_files = glob.glob('debug/latent_vis_for_rebuttal/sample_vis_loco_grid/*.png')

# 读取所有图片
images = [Image.open(file) for file in image_files]

# 计算行数
num_images = len(images)
num_cols = 6
num_rows = math.ceil(num_images / num_cols)

# 假设所有图片大小相同，获取第一张图片的大小
img_width, img_height = images[0].size

# 创建画布
canvas_width = num_cols * img_width
canvas_height = num_rows * img_height
canvas = Image.new('RGB', (canvas_width, canvas_height))

# 将图片粘贴到画布上
for i, img in enumerate(images):
    row = i // num_cols
    col = i % num_cols
    canvas.paste(img, (col * img_width, row * img_height))

# 保存结果
canvas.save('gallery_output.png')