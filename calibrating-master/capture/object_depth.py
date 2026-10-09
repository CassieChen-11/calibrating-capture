#!/usr/bin/env python3
import os
import cv2
import numpy as np
from glob import glob
import calibrating
from calibrating import imread, shows

# ================== 配置区 ==================
# 棋盘格标定图文件夹（原来拍的那 30 组）
CALIB_DIR = r'E:\learning\calibrating\calibrating-master\capture\calibration_images'

# 物体图路径（你刚拍的）
OBJECT_LEFT = r'E:\learning\calibrating\calibrating-master\capture\object_images\1\stereo_l.jpg'
OBJECT_RIGHT = r'E:\learning\calibrating\calibrating-master\capture\object_images\1\stereo_r.jpg'

# 棋盘格参数（和你标定脚本里保持一致）
CHESSBOARD = (8, 11)
SIZE_MM = 5

# 最大深度（米）。目标距离约 1 米，留出近距离余量。
MAX_DEPTH = 1.5
# =============================================

print("步骤1：正在用棋盘格图标定相机...")
board = calibrating.Chessboard(checkboard=CHESSBOARD, size_mm=SIZE_MM)
caml = calibrating.Cam(glob(f"{CALIB_DIR}/*/stereo_l.jpg"), board)
camr = calibrating.Cam(glob(f"{CALIB_DIR}/*/stereo_r.jpg"), board)
print("左相机:", caml)
print("右相机:", camr)

print("步骤2：双目标定...")
stereo = calibrating.Stereo(caml, camr)
print(stereo)

print("步骤3：设置立体匹配...")
stereo_matching = calibrating.SemiGlobalBlockMatching()
stereo.set_stereo_matching(stereo_matching, max_depth=MAX_DEPTH)

print("步骤4：读取物体图并计算深度...")
imgl = imread(OBJECT_LEFT)
imgr = imread(OBJECT_RIGHT)
stereo_result = stereo.get_depth(imgl, imgr)
depth = stereo_result["unrectify_depth"]
undistort_img = stereo_result["undistort_img1"]

# 去除零散误匹配，并填补目标表面的小空洞。
valid = ((depth >= 0.5) & (depth <= MAX_DEPTH)).astype(np.uint8)# 只保留合理深度范围内的像素
kernel = np.ones((5, 5), np.uint8)# 形态学操作的卷积核
valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, kernel)# 闭运算，填补小孔洞
valid = cv2.morphologyEx(valid, cv2.MORPH_OPEN, kernel)# 开运算，去除小噪点

# 进一步去除小连通区域，保留较大的连通区域
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(valid, 8)# 连通组件分析
clean_valid = np.zeros_like(valid)            # 用于存储清理后的有效区域
for label in range(1, num_labels):            # 遍历每个连通区域
	if stats[label, cv2.CC_STAT_AREA] >= 300: # 保留面积大于等于 300 的连通区域
		clean_valid[labels == label] = 1      # 标记为有效区域

filtered_depth = cv2.medianBlur(depth.astype(np.float32), 5) # 中值滤波，去除噪点
depth = np.where(clean_valid, filtered_depth, 0).astype(np.float32)# 仅保留清理后的有效深度

print("步骤5：可视化...")
depth_vis = calibrating.vis_depth(depth, fix_range=(0.5, MAX_DEPTH))# 可视化深度图
shows([depth_vis, undistort_img])
print("Depth map generated; please check the browser.")