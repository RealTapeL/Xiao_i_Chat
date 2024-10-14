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
