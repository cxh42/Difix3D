import os
import shutil

# 原始图片所在的文件夹
source_folder = 'images'
# 用于存放筛选后图片的新文件夹
output_folder = 'images_filtered'

# 如果输出文件夹不存在，创建它
os.makedirs(output_folder, exist_ok=True)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    if filename.endswith('_0.jpg'):
        # 构建完整路径
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(output_folder, filename)
        shutil.copyfile(src_path, dst_path)
        print(f'已复制: {filename}')
