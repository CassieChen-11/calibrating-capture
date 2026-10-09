# -- coding: utf-8 --
"""
华睿双目相机棋盘格采集脚本
基于华睿官方 IMVApi
按键：s=保存  c=角点检测  q=退出
"""
import sys
import os
import cv2
import numpy as np
from ctypes import *
import datetime
from concurrent.futures import ThreadPoolExecutor

# ===== 添加华睿 SDK 路径 =====
SDK_PATH = r"E:\MVviewer\MV Viewer\Development\Samples\Python\IMV\MVSDK"
if SDK_PATH not in sys.path:
    sys.path.append(SDK_PATH)

from IMVApi import *

# ================== 配置区 ==================
# 用序列号区分左右相机（不再依赖索引）
SERIAL_LEFT = "CE00006AAK00003"
SERIAL_RIGHT = "CK83598AAK00015"

# 增益（右边暗就调大 GAIN_RIGHT）
GAIN_LEFT = 20.0
GAIN_RIGHT = 30.0

# 曝光时间（单位：微秒），左右可以不同
EXPOSURE_LEFT = 8000.0
EXPOSURE_RIGHT = 25000.0

# 保存路径
#OUTPUT_ROOT = r"E:\learning\calibrating\calibrating-master\capture\calibration_images"
OUTPUT_ROOT = r"E:\learning\calibrating\calibrating-master\capture\object_images"

# 棋盘格内角点数量 (列, 行)
CHESSBOARD_INNER_CORNERS = (8, 11)
# =============================================


def open_camera_by_index(index, gain, exposure):
    """按设备列表索引打开相机，并设置指定增益"""
    cam = MvCamera()
    idx_c = c_void_p(index)
    nRet = cam.IMV_CreateHandle(IMV_ECreateHandleMode.modeByIndex, byref(idx_c))
    if IMV_OK != nRet:
        print(f"❌ 创建句柄失败! 索引={index}, 错误码={nRet}")
        return None

    nRet = cam.IMV_Open()
    if IMV_OK != nRet:
        print(f"❌ 打开相机失败! 索引={index}, 错误码={nRet}")
        cam.IMV_DestroyHandle()
        return None

    # 关闭触发模式，连续出图
    cam.IMV_SetEnumFeatureSymbol("TriggerSource", "Software")
    cam.IMV_SetEnumFeatureSymbol("TriggerSelector", "FrameStart")
    cam.IMV_SetEnumFeatureSymbol("TriggerMode", "Off")

    # 关闭自动曝光/自动增益
    ret_a = cam.IMV_SetEnumFeatureSymbol("ExposureAuto", "Off")
    ret_b = cam.IMV_SetEnumFeatureSymbol("GainAuto", "Off")
    print(f"  [索引{index}] 关闭自动曝光返回={ret_a}, 关闭自动增益返回={ret_b}")

    # 用 Double 类型函数设置曝光和增益
    ret_e = cam.IMV_SetDoubleFeatureValue("ExposureTime", float(exposure))
    ret_g = cam.IMV_SetDoubleFeatureValue("Gain", float(gain))
    print(f"  [索引{index}] 设置曝光返回={ret_e}, 设置增益返回={ret_g}")

    # 读回来验证
    exp_val = c_double()
    gain_val = c_double()
    cam.IMV_GetDoubleFeatureValue("ExposureTime", exp_val)
    cam.IMV_GetDoubleFeatureValue("Gain", gain_val)
    print(f"  [索引{index}] 实际曝光={exp_val.value} 微秒, 实际增益={gain_val.value}")

    print(f"✅ 已打开相机，索引={index}, Gain={gain}")

    return cam


def get_frame_bgr(cam, timeout_ms=1000):
    """从相机获取一帧并转换为BGR图像"""
    frame = IMV_Frame()
    nRet = cam.IMV_GetFrame(frame, timeout_ms)
    if IMV_OK != nRet:
        return None

    # 分配目标缓冲区
    if frame.frameInfo.pixelFormat == IMV_EPixelType.gvspPixelMono8:
        nDstBufSize = frame.frameInfo.width * frame.frameInfo.height
    else:
        nDstBufSize = frame.frameInfo.width * frame.frameInfo.height * 3

    pDstBuf = (c_ubyte * nDstBufSize)()

    stParam = IMV_PixelConvertParam()
    memset(byref(stParam), 0, sizeof(stParam))
    stParam.nWidth = frame.frameInfo.width
    stParam.nHeight = frame.frameInfo.height
    stParam.ePixelFormat = frame.frameInfo.pixelFormat
    stParam.pSrcData = frame.pData
    stParam.nSrcDataLen = frame.frameInfo.size
    stParam.nPaddingX = frame.frameInfo.paddingX
    stParam.nPaddingY = frame.frameInfo.paddingY
    stParam.eBayerDemosaic = IMV_EBayerDemosaic.demosaicEdgeSensing
    stParam.eDstPixelFormat = IMV_EPixelType.gvspPixelBGR8
    stParam.pDstBuf = pDstBuf
    stParam.nDstBufSize = nDstBufSize

    nRet = cam.IMV_PixelConvert(stParam)
    if IMV_OK != nRet:
        print(f"❌ 像素格式转换失败! 错误码={nRet}")
        cam.IMV_ReleaseFrame(frame)
        return None

    # 释放驱动缓存
    cam.IMV_ReleaseFrame(frame)

    # 构造 numpy 图像
    buf = c_buffer(b'\0', stParam.nDstBufSize)
    memmove(buf, stParam.pDstBuf, stParam.nDstBufSize)
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = arr.reshape((stParam.nHeight, stParam.nWidth, 3))
    return img

#并发获取左右相机的一对图像
def get_stereo_pair(executor, cam_left, cam_right):
    """Concurrently acquire one frame from each camera."""
    left_future = executor.submit(get_frame_bgr, cam_left)
    right_future = executor.submit(get_frame_bgr, cam_right)
    return left_future.result(), right_future.result()

# 根据序列号查找相机索引
def find_index_by_serial(deviceList, serial):
    """根据序列号在设备列表中查找 0-based 索引"""
    for i in range(deviceList.nDevNum):
        info = deviceList.pDevInfo[i]
        sn = info.serialNumber.decode("utf-8")
        if sn == serial:
            return i
    return -1

def main():
    # 1. 枚举设备
    deviceList = IMV_DeviceList()
    nRet = MvCamera.IMV_EnumDevices(deviceList, IMV_EInterfaceType.interfaceTypeAll)
    if IMV_OK != nRet or deviceList.nDevNum < 2:
        print(f"❌ 枚举设备失败或设备不足2台! 错误码={nRet}, 设备数={deviceList.nDevNum}")
        return

    print(f"找到 {deviceList.nDevNum} 个设备")

    for i in range(deviceList.nDevNum):
        info = deviceList.pDevInfo[i]
        sn = info.serialNumber.decode("utf-8")
        print(f"  设备[{i}] 序列号: {sn}")

        # 2. 用序列号查找左右相机索引
    idx_left = find_index_by_serial(deviceList, SERIAL_LEFT)
    idx_right = find_index_by_serial(deviceList, SERIAL_RIGHT)
    if idx_left < 0 or idx_right < 0:
        print(f"❌ 未找到指定序列号的相机! 左={SERIAL_LEFT}, 右={SERIAL_RIGHT}")
        return
    print(f"左相机索引={idx_left}, 右相机索引={idx_right}")

    # 3. 打开左右相机
    cam_left = open_camera_by_index(idx_left, GAIN_LEFT, EXPOSURE_LEFT)
    cam_right = open_camera_by_index(idx_right, GAIN_RIGHT, EXPOSURE_RIGHT)
    if cam_left is None or cam_right is None:
        return

    # 3. 开始拉流
    cam_left.IMV_StartGrabbing()
    cam_right.IMV_StartGrabbing()
    frame_executor = ThreadPoolExecutor(max_workers=2)

    # 4. 准备保存目录
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    saved_count = 0
    show_corners = False
    window_name = "Huaray Stereo Capture (s:save c:corner q:quit)"

    print("\n=== 双目采集已启动 ===")
    print("按 s 保存当前一对图像")
    print("按 c 切换角点检测辅助显示")
    print("按 q 退出\n")

    try:
        while True:
            # 5. 并发取左右帧，避免先取左帧再取右帧造成额外时间差
            img_left, img_right = get_stereo_pair(
                frame_executor, cam_left, cam_right
            )

            if img_left is None or img_right is None:
                print("⚠️ 取帧失败，继续重试...")
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue

            # 6. 高度对齐
            h = min(img_left.shape[0], img_right.shape[0])
            img_left = img_left[:h, :]
            img_right = img_right[:h, :]

            # 7. 角点检测辅助显示
            if show_corners:
                # 复制一份用于显示，避免污染原始图像
                img_left_disp = img_left.copy()
                img_right_disp = img_right.copy()
                gray_l = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
                ok_l, cor_l = cv2.findChessboardCorners(gray_l, CHESSBOARD_INNER_CORNERS, None)
                ok_r, cor_r = cv2.findChessboardCorners(gray_r, CHESSBOARD_INNER_CORNERS, None)
                if ok_l:
                    cv2.drawChessboardCorners(img_left_disp, CHESSBOARD_INNER_CORNERS, cor_l, ok_l)
                if ok_r:
                    cv2.drawChessboardCorners(img_right_disp, CHESSBOARD_INNER_CORNERS, cor_r, ok_r)
                status = f"Corners L/R: {'OK' if ok_l else 'NG'} / {'OK' if ok_r else 'NG'}"
            else:
                img_left_disp = img_left
                img_right_disp = img_right
                status = f"Saved: {saved_count}  (press c for corners)"

            # 8. 拼接预览
            preview = np.hstack((img_left_disp, img_right_disp))
            cv2.putText(preview, status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # 仅用于显示：按比例缩小预览图，避免超出屏幕
            scale = 0.5  # 屏幕够大就改成 0.7 或 1.0
            preview_disp = cv2.resize(preview, (int(preview.shape[1] * scale),
                                    int(preview.shape[0] * scale)))
            cv2.imshow(window_name, preview_disp)
            # cv2.imshow(window_name, preview)  # 原始大小预览，已用缩放预览替代

            # 9. 按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                show_corners = not show_corners
                print(f"角点检测: {'开启' if show_corners else '关闭'}")
            elif key == ord('s'):
                saved_count += 1
                group_dir = os.path.join(OUTPUT_ROOT, str(saved_count))
                os.makedirs(group_dir, exist_ok=True)
                cv2.imwrite(os.path.join(group_dir, "stereo_l.jpg"),
                            img_left, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(os.path.join(group_dir, "stereo_r.jpg"),
                            img_right, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"✅ 已保存第 {saved_count} 组到 {group_dir}")

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        frame_executor.shutdown(wait=True)
        cam_left.IMV_StopGrabbing()
        cam_left.IMV_Close()
        cam_left.IMV_DestroyHandle()

        cam_right.IMV_StopGrabbing()
        cam_right.IMV_Close()
        cam_right.IMV_DestroyHandle()

        cv2.destroyAllWindows()
        print("相机已释放，程序结束")


if __name__ == "__main__":
    main()