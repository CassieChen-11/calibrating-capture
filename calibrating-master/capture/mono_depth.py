#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用 MiDaS 单目深度估计模型生成深度图"""
import cv2
import torch
import numpy as np

# 配置
IMAGE_PATH = r"E:\learning\calibrating\calibrating-master\capture\object_images\1\stereo_l.jpg"
OUTPUT_PATH = r"E:\learning\calibrating\calibrating-master\capture\mono_depth_result.png"

# 加载 MiDaS 模型
print("正在加载 MiDaS 模型...")
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
model.eval()

# 选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"模型后端: {device}")

# 读取图像
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"读不到图片: {IMAGE_PATH}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 预处理
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
input_tensor = transform(img_rgb).to(device)

# 推理
print("正在推理...")
with torch.no_grad():
    depth = model(input_tensor)

# 后处理：恢复原图尺寸
depth = torch.nn.functional.interpolate(
    depth.unsqueeze(1),
    size=img.shape[:2],
    mode="bicubic",
    align_corners=False,
).squeeze().cpu().numpy()

# 可视化
depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
cv2.imwrite(OUTPUT_PATH, depth_color)
print(f"深度图已保存: {OUTPUT_PATH}")

# 同时显示
cv2.imshow("Mono Depth", depth_color)
cv2.waitKey(0)
cv2.destroyAllWindows()