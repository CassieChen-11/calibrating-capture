#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用 RAFT-Stereo 预训练模型进行双目深度估计。

默认流程：
1. 从 stereo_calibration.yaml 读取 calibrating 标定结果；
2. 如果 YAML 不存在，则使用棋盘格图片重新标定并保存 YAML；
3. 对左右原始图片进行立体校正；
4. 使用 RAFT-Stereo PyTorch 模型计算稠密视差；
5. 使用焦距和 baseline 将视差转换为米制深度；
6. 保存视差、深度和伪彩色结果。

首次运行前请安装：
    pip install opencv-python numpy pyyaml pillow
    pip install torch torchvision    # 按 PyTorch 官网选择 CPU/CUDA 版本

RAFT-Stereo 源码和 Middlebury 预训练权重会自动下载到 models/raft_stereo/。
"""

from glob import glob
import os
import sys
import time
import urllib.request
import zipfile
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import calibrating
from calibrating import shows

# 兼容 calibrating/boxx 旧代码与新版 NumPy 的类型别名变化。
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_


# ============================== 配置 ==============================
ROOT = Path(__file__).resolve().parents[1]
CALIB_DIR = ROOT / "capture" / "calibration_images"
LEFT_IMAGE = ROOT / "capture" / "object_images" / "1" / "stereo_l.jpg"#手柄
RIGHT_IMAGE = ROOT / "capture" / "object_images" / "1" / "stereo_r.jpg"
#LEFT_IMAGE = ROOT / "capture" / "object_images" / "2" / "stereo_l.jpg"#维生素瓶
#RIGHT_IMAGE = ROOT / "capture" / "object_images" / "2" / "stereo_r.jpg"
CALIBRATION_FILE = ROOT / "capture" / "stereo_calibration.yaml"
MODEL_DIR = ROOT / "models"
RAFT_DIR = MODEL_DIR / "raft_stereo"
RAFT_REPO_URL = "https://github.com/princeton-vl/RAFT-Stereo/archive/refs/heads/main.zip"
RAFT_MODELS_URL = "https://www.dropbox.com/scl/fi/5khx1bhz84dapi8vtwapg/models.zip?rlkey=ggddrn1du1iiq6mgc2dsdpmwi&dl=1"
MODEL_FILE = RAFT_DIR / "models" / "raftstereo-middlebury.pth"

CHESSBOARD = (8, 11)    # 棋盘格内角点数量 (行, 列)
SQUARE_SIZE_MM = 5      # 棋盘格方格边长，单位毫米
MIN_DEPTH = 0.6          # 最小深度（米）
MAX_DEPTH = 1.0          # 最大深度（米）
INFERENCE_SIZE = (1024, 1280)  # (height, width)，与当前相机图像一致，保留轮廓细节
USE_GPU = True          # 是否使用 GPU 进行推理
VALID_ITERS = 24          # 有效迭代次数
SAVE_RECTIFIED = True          # 是否保存立体校正后的图像
ENABLE_DEPTH_INSPECTOR = True          # 是否启用深度检查器
ENABLE_BROWSER_VIEW = True          # 是否启用浏览器视图
# 手柄区域 (x0, y0, x1, y1)，用于点云提取和显示放大（像素坐标）。
INSPECT_ROI = (650, 420, 1270, 1024)
# 手柄外轮廓的粗略多边形，坐标是原图像素；用于排除左侧背景（像素坐标）。
HANDLE_POLYGON = np.array(
    [[730, 465], [850, 445], [1010, 445], [1140, 465],
     [1215, 540], [1245, 720], [1245, 875], [1215, 970],
     [1160, 1010], [1090, 1015], [1035, 965], [1000, 900],
     [960, 845], [920, 810], [880, 845], [835, 900],
     [800, 965], [750, 1010], [690, 980], [675, 900],
     [680, 760], [700, 590]],
    dtype=np.int32,
)
# 手柄主体内部的一个像素，用于选择正确的 GrabCut 连通区域（像素坐标）。
POINT_CLOUD_SEED = (950, 650)#手柄
#POINT_CLOUD_SEED = (681, 611)#维生素瓶
ENABLE_POINT_CLOUD = False
# ==================================================================


def download_raft_files():
    """下载官方 RAFT-Stereo 源码和 checkpoint。"""
    RAFT_DIR.mkdir(parents=True, exist_ok=True)
    source_root = RAFT_DIR / "RAFT-Stereo-main"
    if not source_root.exists():
        archive = RAFT_DIR / "raft_stereo_source.zip"
        print("正在下载 RAFT-Stereo 源码...")
        urllib.request.urlretrieve(RAFT_REPO_URL, archive)
        with zipfile.ZipFile(archive) as package:
            package.extractall(RAFT_DIR)
        archive.unlink(missing_ok=True)

    if not MODEL_FILE.exists() or MODEL_FILE.stat().st_size < 10_000_000:
        archive = RAFT_DIR / "models.zip"
        print("正在下载 RAFT-Stereo 预训练权重...")
        urllib.request.urlretrieve(RAFT_MODELS_URL, archive)
        with zipfile.ZipFile(archive) as package:
            package.extractall(RAFT_DIR)
        archive.unlink(missing_ok=True)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"权重下载完成，但找不到 {MODEL_FILE}")


def build_or_load_calibration():
    """优先读取已有标定，缺失时按现有棋盘格图片重新标定。"""
    if CALIBRATION_FILE.exists():
        print(f"读取标定文件: {CALIBRATION_FILE}")
        return calibrating.Stereo().load(str(CALIBRATION_FILE))

    left_paths = sorted(glob(str(CALIB_DIR / "*" / "stereo_l.jpg")))
    right_paths = sorted(glob(str(CALIB_DIR / "*" / "stereo_r.jpg")))
    if not left_paths or not right_paths:
        raise FileNotFoundError(f"找不到标定图片: {CALIB_DIR}")

    print(f"重新标定: 左图 {len(left_paths)} 张，右图 {len(right_paths)} 张")
    board = calibrating.Chessboard(
        checkboard=CHESSBOARD,
        size_mm=SQUARE_SIZE_MM,
    )
    cam_left = calibrating.Cam(
        left_paths,
        board,
        save_feature_vis=False,
    )
    cam_right = calibrating.Cam(
        right_paths,
        board,
        save_feature_vis=False,
    )
    stereo = calibrating.Stereo(cam_left, cam_right)
    stereo.dump(str(CALIBRATION_FILE))
    print(f"标定已保存: {CALIBRATION_FILE}")
    print(f"双目标定 RMS: {stereo.retval:.4f} px")
    return stereo


def load_network():
    """加载官方 RAFT-Stereo 网络和 checkpoint。"""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "当前虚拟环境没有 PyTorch，请先按 PyTorch 官网命令安装 torch。"
        ) from error

    download_raft_files()
    source_root = RAFT_DIR / "RAFT-Stereo-main"
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(source_root / "core"))
    from core.raft_stereo import RAFTStereo

    device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
    args = Namespace(
        hidden_dims=[128, 128, 128],
        context_norm="instance",
        corr_implementation="alt",
        corr_levels=4,
        corr_radius=4,
        n_downsample=2,
        n_gru_layers=3,
        shared_backbone=False,
        slow_fast_gru=False,
        mixed_precision=device == "cuda",
    )
    model = RAFTStereo(args)
    checkpoint = torch.load(MODEL_FILE, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if any(key.startswith("module.") for key in checkpoint):
        checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}
    # 获取模型自身认识的键
    model_keys = set(model.state_dict().keys())
    # 只保留模型认识的键，丢弃多余的（如 cnet 的 running_mean 等）
    filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_keys}
    print(f"原始权重键数: {len(checkpoint)}，过滤后: {len(filtered_checkpoint)}")
    missing, unexpected = model.load_state_dict(filtered_checkpoint, strict=False)
    print(f"缺失的键: {len(missing)} 个")
    print(f"多余的键: {len(unexpected)} 个（已忽略）")
    model.to(device).eval()
    print(f"模型后端: {device}")
    return model, device


def pad_to_size(image, size):
    """缩放并补边到 RAFT-Stereo 输入尺寸，返回缩放比例。"""
    import torch

    target_height, target_width = size
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)))
    canvas = np.zeros((target_height, target_width, 3), np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    tensor = torch.from_numpy(canvas[:, :, ::-1].copy()).permute(2, 0, 1).float()
    return tensor[None], scale, resized.shape[:2]


# ============================== RAFT-Stereo 推理 ==============================
def predict_disparity(model, device, left, right):
    """运行 RAFT-Stereo，输出原始校正图分辨率下的视差。"""
    import torch

    left_tensor, scale, resized_shape = pad_to_size(left, INFERENCE_SIZE)# 缩放并补边到 RAFT-Stereo 输入尺寸
    right_tensor, _, _ = pad_to_size(right, INFERENCE_SIZE)# 缩放并补边到 RAFT-Stereo 输入尺寸
    left_tensor = left_tensor.to(device)# 移动到指定设备
    right_tensor = right_tensor.to(device)# 移动到指定设备
    with torch.no_grad():# 禁用梯度计算，节省显存和计算资源
        _, prediction = model(
            left_tensor,
            right_tensor,
            iters=VALID_ITERS,
            test_mode=True,
        )
    # 官方 demo 保存 -flow_up；左图到右图的水平流为负视差。
    disparity = -prediction[0, 0].detach().float().cpu().numpy()
    disparity = disparity[: resized_shape[0], : resized_shape[1]]
    disparity = cv2.resize(disparity, (left.shape[1], left.shape[0])) / scale
    return disparity


def resize_disparity_to_original(disparity, original_shape):
    """恢复视差尺寸，并按水平方向缩放视差数值。"""
    original_height, original_width = original_shape[:2]
    model_height, model_width = disparity.shape
    restored = cv2.resize(
        disparity,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )
    restored *= original_width / float(model_width)
    return restored


def disparity_to_depth(disparity, stereo):
    """利用 calibrating 的 baseline 和校正后焦距计算米制深度。"""
    focal_length = float(stereo.K[0, 0])
    baseline = float(stereo.baseline)
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 0.1)
    depth[valid] = focal_length * baseline / disparity[valid]
    depth[(depth < MIN_DEPTH) | (depth > MAX_DEPTH)] = 0
    return depth


def save_handle_point_cloud(left_image, depth, stereo, output_dir):
    """用 RGB 分割和深度范围提取手柄 XYZRGB 点云，坐标单位为米。"""
    x0, y0, x1, y1 = INSPECT_ROI
    height, width = depth.shape
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"INSPECT_ROI 无效: {INSPECT_ROI}")

    depth_roi = depth[y0:y1, x0:x1]
    color_roi = left_image[y0:y1, x0:x1]
    depth_valid = np.isfinite(depth_roi) & (depth_roi > MIN_DEPTH) & (depth_roi < MAX_DEPTH)

    # 多边形外强制设为背景，避免 GrabCut 把左侧桌面当成前景。
    grabcut_mask = np.full(depth_roi.shape, cv2.GC_BGD, np.uint8)
    polygon = HANDLE_POLYGON - np.array([x0, y0], dtype=np.int32)
    cv2.fillPoly(grabcut_mask, [polygon], cv2.GC_PR_FGD)
    seed_x, seed_y = POINT_CLOUD_SEED[0] - x0, POINT_CLOUD_SEED[1] - y0
    if not (0 <= seed_x < grabcut_mask.shape[1] and 0 <= seed_y < grabcut_mask.shape[0]):
        raise ValueError(f"POINT_CLOUD_SEED 必须位于 INSPECT_ROI 内: {POINT_CLOUD_SEED}")
    cv2.circle(grabcut_mask, (seed_x, seed_y), 35, cv2.GC_FGD, -1)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(color_roi, grabcut_mask, None, bg_model, fg_model, 8, cv2.GC_INIT_WITH_MASK)
    rgb_valid = (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD)

    # 深度无效点不能进入点云；形态学操作只清理边缘毛刺，不扩张背景。
    valid = rgb_valid & depth_valid
    valid_u8 = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    valid_u8 = cv2.morphologyEx(valid_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(valid_u8, 8)
    seed_label = labels[seed_y, seed_x]
    if seed_label == 0:
        raise RuntimeError("手柄种子点没有落在有效深度区域，请调整 POINT_CLOUD_SEED")
    valid = labels == seed_label

    yy, xx = np.indices(depth_roi.shape, dtype=np.float32)
    xx += x0
    yy += y0
    fx, fy = float(stereo.K[0, 0]), float(stereo.K[1, 1])
    cx, cy = float(stereo.K[0, 2]), float(stereo.K[1, 2])
    z = depth_roi[valid].astype(np.float32)
    x = (xx[valid] - cx) * z / fx
    y = (yy[valid] - cy) * z / fy
    points = np.column_stack((x, y, z)).astype(np.float32)
    colors = color_roi[valid][:, ::-1].astype(np.uint8)  # BGR -> RGB

    np.save(output_dir / "handle_point_cloud_xyzrgb.npy", np.column_stack((points, colors)))
    mask_image = np.zeros_like(left_image)
    mask_image[y0:y1, x0:x1][valid] = color_roi[valid]
    cv2.circle(mask_image, POINT_CLOUD_SEED, 8, (0, 255, 255), 2)
    cv2.imwrite(str(output_dir / "handle_point_cloud_mask.png"), mask_image)

    ply_path = output_dir / "handle_point_cloud.ply"
    with ply_path.open("w", encoding="ascii") as ply:
        ply.write("ply\nformat ascii 1.0\n")
        ply.write(f"element vertex {len(points)}\n")
        ply.write("property float x\nproperty float y\nproperty float z\n")
        ply.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        ply.write("end_header\n")
        for point, color in zip(points, colors):
            ply.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
    print(f"手柄点云已保存: {ply_path}，点数: {len(points)}")


def apply_rectification_mask(disparity, stereo):
    """移除立体校正产生的黑边，避免黑色无效区域被当成深度。"""
    valid_mask = getattr(stereo, "rectify_valid_mask1", None)
    if valid_mask is None:
        return disparity
    if valid_mask.shape != disparity.shape:
        raise ValueError(
            f"校正掩码尺寸 {valid_mask.shape} 与视差尺寸 {disparity.shape} 不一致"
        )
    return np.where(valid_mask, disparity, 0).astype(np.float32)


def save_color_visualization(array, output_path, value_range):
    """保存带固定范围的 JET 伪彩色图，零值显示为黑色。"""
    low, high = value_range
    normalized = np.clip((array - low) / (high - low), 0, 1)
    color = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_JET)
    color[array <= 0] = 0
    cv2.imwrite(str(output_path), color)


def save_detail_visualization(left_image, depth, output_dir):
    """保存高对比度深度图，并用左图边缘辅助观察真实轮廓。"""
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return

    low, high = np.percentile(depth[valid], (2, 98))
    if high <= low:
        low, high = MIN_DEPTH, MAX_DEPTH
    normalized = np.clip((depth - low) / (high - low), 0, 1)
    detail = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_TURBO)
    detail[~valid] = 0

    gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    detail[edges > 0] = (255, 255, 255)
    cv2.imwrite(str(output_dir / "depth_detail_edges.png"), detail)
    print(f"深度细节显示范围: {low:.3f} ~ {high:.3f} m")


def make_inspector_depth(depth):
    """用深度分位数拉伸颜色，并叠加深度梯度边界。"""
    valid = np.isfinite(depth) & (depth > 0)
    enhanced = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return enhanced

    low, high = np.percentile(depth[valid], (1, 99))
    if high <= low:
        low, high = MIN_DEPTH, MAX_DEPTH
    normalized = np.clip((depth - low) / (high - low), 0, 1)
    enhanced = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_TURBO)
    enhanced[~valid] = 0

    depth_gray = np.uint8(normalized * 255)
    gradient_x = cv2.Sobel(depth_gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(depth_gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    edge_threshold = max(float(np.percentile(gradient[valid], 92)), 25.0)
    depth_edges = (gradient >= edge_threshold) & valid
    depth_edges = cv2.morphologyEx(
        depth_edges.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    enhanced[depth_edges > 0] = (255, 255, 255)
    return enhanced


def save_inspector_zoom(left_image, depth, output_dir):
    """保存目标区域放大图，避免整幅图缩小后看不清轮廓。"""
    height, width = depth.shape
    x0, y0, x1, y1 = INSPECT_ROI
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return

    depth_roi = make_inspector_depth(depth[y0:y1, x0:x1])
    rgb_roi = left_image[y0:y1, x0:x1].copy()
    gray = cv2.cvtColor(rgb_roi, cv2.COLOR_BGR2GRAY)
    rgb_edges = cv2.Canny(gray, 40, 120)
    rgb_overlay = rgb_roi.copy()
    rgb_overlay[rgb_edges > 0] = (0, 255, 255)
    zoom = cv2.resize(
        np.hstack((rgb_overlay, depth_roi)),
        (0, 0),
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imwrite(str(output_dir / "depth_inspection_zoom.png"), zoom)


def inspect_depth(left_image, depth, output_dir):
    """交互读取深度：点击左右任一图像，显示邻域深度中位数。"""
    height, width = depth.shape
    inspector_depth = make_inspector_depth(depth)
    save_inspector_zoom(left_image, depth, output_dir)
    full_panel = np.hstack((left_image.copy(), inspector_depth))
    display_scale = min(1.0, 1500.0 / full_panel.shape[1], 850.0 / full_panel.shape[0])
    if display_scale < 1.0:
        panel = cv2.resize(
            full_panel,
            (round(full_panel.shape[1] * display_scale), round(full_panel.shape[0] * display_scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        panel = full_panel.copy()
    window_name = "Depth Inspector (click to measure, q/Esc to exit)"

    def on_mouse(event, x, y, _flags, _param):
        nonlocal panel
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        original_x = round(x / display_scale)
        original_y = round(y / display_scale)
        display_width = round(width * display_scale)
        image_x = original_x if original_x < width else original_x - width
        if not (0 <= image_x < width and 0 <= original_y < height):
            return

        radius = 5
        y0, y1 = max(0, original_y - radius), min(height, original_y + radius + 1)
        x0, x1 = max(0, image_x - radius), min(width, image_x + radius + 1)
        neighborhood = depth[y0:y1, x0:x1]
        valid = neighborhood[np.isfinite(neighborhood) & (neighborhood > 0)]
        value = float(np.median(valid)) if valid.size else 0.0

        marked = full_panel.copy()
        cv2.drawMarker(marked, (image_x, original_y), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.drawMarker(marked, (image_x + width, original_y), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
        label = f"x={image_x}, y={original_y}, depth={value:.3f} m"
        cv2.rectangle(marked, (10, 10), (500, 58), (0, 0, 0), -1)
        cv2.putText(
            marked,
            label,
            (20, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if display_scale < 1.0:
            panel = cv2.resize(
                marked,
                (round(marked.shape[1] * display_scale), round(marked.shape[0] * display_scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            panel = marked
        cv2.imwrite(str(output_dir / "depth_inspection.png"), marked)
        if value > 0:
            print(f"点击 ({image_x}, {original_y}) -> 深度 {value:.3f} m")
        else:
            print(f"点击 ({image_x}, {original_y}) -> 无有效深度")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    print("深度检查器已打开：点击图像读取深度，按 q 或 Esc 退出。")
    while True:
        cv2.imshow(window_name, panel)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
    cv2.destroyWindow(window_name)


def main():
    if not LEFT_IMAGE.exists() or not RIGHT_IMAGE.exists():
        raise FileNotFoundError("找不到左右目标图片，请检查 LEFT_IMAGE 和 RIGHT_IMAGE")

    stereo = build_or_load_calibration()
    model, device = load_network()

    left = cv2.imread(str(LEFT_IMAGE), cv2.IMREAD_COLOR)
    right = cv2.imread(str(RIGHT_IMAGE), cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise RuntimeError("左右图片读取失败")

    print(f"原始图像尺寸: {left.shape[1]}x{left.shape[0]}")
    rectified_left, rectified_right = stereo.rectify(left, right)

    start = time.perf_counter()
    model_disparity = predict_disparity(
        model,
        device,
        rectified_left,
        rectified_right,
    )
    elapsed = time.perf_counter() - start
    disparity = resize_disparity_to_original(model_disparity, rectified_left.shape)
    disparity = apply_rectification_mask(disparity, stereo)
    depth = disparity_to_depth(disparity, stereo)

    output_dir = ROOT / "output_deep_stereo"#手柄
    #output_dir = ROOT / "output_deep_stereo_2"#维生素瓶
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "disparity.npy", disparity)
    np.save(output_dir / "depth_m.npy", depth)
    if SAVE_RECTIFIED:
        cv2.imwrite(str(output_dir / "rectified_left.png"), rectified_left)
        cv2.imwrite(str(output_dir / "rectified_right.png"), rectified_right)

    save_color_visualization(
        disparity,
        output_dir / "disparity_color.png",
        (0.0, max(float(np.percentile(disparity[disparity > 0], 99)), 1.0)),
    )
    save_color_visualization(
        depth,
        output_dir / "depth_color.png",
        (MIN_DEPTH, MAX_DEPTH),
    )
    save_detail_visualization(rectified_left, depth, output_dir)
    depth_color = cv2.imread(str(output_dir / "depth_color.png"))
    if ENABLE_POINT_CLOUD:
        save_handle_point_cloud(rectified_left, depth, stereo, output_dir)

    valid_disparity = disparity[np.isfinite(disparity) & (disparity > 0)]
    valid_depth = depth[depth > 0]
    print(f"baseline: {stereo.baseline:.6f} m")
    print(f"校正后 fx: {stereo.K[0, 0]:.3f} px")
    print(f"模型推理耗时: {elapsed:.3f} s")
    print(f"有效视差比例: {len(valid_disparity) / disparity.size:.2%}")
    print(f"有效深度比例: {len(valid_depth) / depth.size:.2%}")
    if len(valid_depth):
        print(
            "深度范围: "
            f"{valid_depth.min():.3f} ~ {valid_depth.max():.3f} m，"
            f"中位数 {np.median(valid_depth):.3f} m"
        )
    print(f"结果已保存到: {output_dir}")
    if ENABLE_BROWSER_VIEW:
        print("正在打开浏览器预览...")
        shows([
            cv2.cvtColor(rectified_left, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(cv2.imread(str(output_dir / "handle_point_cloud_mask.png")), cv2.COLOR_BGR2RGB),
        ])
    if ENABLE_DEPTH_INSPECTOR:
        inspect_depth(rectified_left, depth, output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"运行失败: {error}", file=sys.stderr)
        raise
