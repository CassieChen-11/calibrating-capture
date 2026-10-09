#!/usr/bin/env python3

import os
import sys
from glob import glob
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 兼容 calibrating/boxx 旧代码与新版 NumPy 的类型别名变化。
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import calibrating  # pip install calibrating
from calibrating import imread, shows

checkboard_img_dir = r'E:\learning\calibrating\calibrating-master\capture\calibration_images'

assert os.path.isdir(
    checkboard_img_dir
), '图片路径不存在，请检查 checkboard_img_dir'

# ▮ 标定所有相机的内参
# 准备棋盘格特征提取器
board = calibrating.Chessboard(checkboard=(8, 11), size_mm=5)

# 实例化的过程中会自动标定内参 cam.K, cam.D, 并在 `/tmp/calibrating-*` 下存放 feature 可视化
# 左目/右目
caml = calibrating.Cam(glob(f"{checkboard_img_dir}/*/stereo_l.jpg"), board)
camr = calibrating.Cam(glob(f"{checkboard_img_dir}/*/stereo_r.jpg"), board)

print(caml, camr)
# 用浏览器可视化在相机视野中每一块标定板的 image_point 和 Rt
shows([cam.vis_image_points_cover() for cam in (caml, camr)])

# ▮ 标定双目相机
stereo = calibrating.Stereo(caml, camr)
print(stereo)
# 左右图像 rectify 对齐可视化
caml.vis_stereo(camr, stereo)
# depth 精度理论分析
# stereo.precision_analysis()

# ▮ 使用 stereo_matching 算法(SemiGlobalBlockMatching) 来计算深度
max_depth = 1.2
stereo_matching = calibrating.SemiGlobalBlockMatching()
stereo.set_stereo_matching(stereo_matching, max_depth=max_depth)
# 准备一对左右相机的图像
key = caml.valid_keys_intersection(camr)[0]
imgl = imread(caml[key]["path"])
imgr = imread(camr[key]["path"])
stereo_result = stereo.get_depth(imgl, imgr)
depth_stereo = stereo_result["unrectify_depth"]
# 1~3.5m 内的绝对深度可视化
depth_stereo_vis = calibrating.vis_depth(depth_stereo, fix_range=(0.4, max_depth))
undistort_img1 = stereo_result["undistort_img1"]

# 浏览器中可视化图像, 注意:
#   - 所有的深度都是和去畸变后的左相机图像对齐
#   - 标定图像最好超过 30 张
shows([depth_stereo_vis, undistort_img1])