# 使用PaddleOCR批量处理识别PDF文档
## 1、简介
本项目利用PaddleOCR库实现对PDF文件中的文字进行批量识别和处理。PaddleOCR是一个基于PaddlePaddle深度学习框架开发的开源字符识别（OCR）工具库，具备高效、准确、易用的特点。
## 2. 环境准备
### 2.1 安装PaddlePaddle
首先，确保已经安装了PaddlePaddle深度学习框架。可以通过pip进行安装。注意：根据你的操作系统和硬件环境（如是否使用GPU），可能需要安装特定版本的PaddlePaddle。请参考PaddlePaddle官方文档进行安装。
`pip install paddlepaddle`
### 2.2 安装PaddleOCR
`pip install paddleocr`
## 3. PDF文件预处理
由于PaddleOCR主要处理图像数据，因此我们需要将PDF文件转换为图像格式。可以使用一些PDF处理库（如PyMuPDF、PDFMiner等）将PDF文件中的每一页转换为图像文件，并保存为单独的图片。
## 4. 批量识别处理
### 4.1 加载PaddleOCR模型
首先，导入PaddleOCR库并加载预训练模型：
```
from paddleocr import PaddleOCR, draw_ocr
#初始化OCR模型，使用默认的英文模型
ocr = PaddleOCR(use_angle_cls=True, lang='en')
```
>注意：`lang`参数用于指定识别语言，可以根据需要选择相应的语言模型。
### 4.2 批量识别PDF图像
编写代码遍历所有转换后的PDF图像文件，并使用PaddleOCR进行文字识别：
```
import os
import glob

# 假设PDF图像文件保存在"pdf_images"文件夹中
image_dir = "pdf_images"
image_list = glob.glob(os.path.join(image_dir, "*.jpg"))  # 根据实际情况修改文件扩展名

results = []
for img_path in image_list:
    # 读取图像
    img = cv2.imread(img_path)
    
    # 使用OCR模型进行识别
    result = ocr.ocr(img, use_gpu=False)
    
    # 将识别结果添加到列表中
    results.append((img_path, result))
```
### 4.3 处理识别结果
识别结果`result`是一个列表，其中每个元素是一个包含文本信息和位置信息的元组。可以根据需要对这些结果进行进一步处理，如提取文本、保存为文件等。
## 6. 示例代码
以下是一个简单的示例代码，展示了如何使用PaddleOCR对PDF文件进行批量识别处理：
```
import cv2
from paddleocr import PaddleOCR, draw_ocr
import os
import glob

# 初始化OCR模型
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# 设置PDF图像文件夹路径
image_dir = "pdf_images"

# 获取PDF图像文件列表
image_list = glob.glob(os.path.join(image_dir, "*.jpg"))  # 根据实际情况修改文件扩展名

# 批量识别处理
results = []
for img_path in image_list:
    # 读取图像
    img = cv2.imread(img_path)
    
    # 使用OCR模型进行识别
    result = ocr.ocr(img, use_gpu=False)
    
    # 可视化识别结果（可选）
    # image_show = draw_ocr(img, result, font_path='./doc/fonts/simfang.ttf')
    # cv2.imshow('ocr_result', image_show)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # 将识别结果添加到列表中
    results.append((img_path, result))

# 处理识别结果
```
