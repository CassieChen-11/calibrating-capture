import cv2

img_path = r"E:\learning\calibrating\calibrating-master\capture\calibration_images\1\stereo_l.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(f"图像尺寸: {gray.shape}")

# 尝试多种尺寸 + 宽松标志
flags_list = [
    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS,
    cv2.CALIB_CB_ADAPTIVE_THRESH,
]

sizes = [(7,10),(10,7),(8,11),(11,8),(6,9),(9,6),(7,9),(9,7),(8,10),(10,8),(9,12),(12,9)]

for flags in flags_list:
    for size in sizes:
        ret, corners = cv2.findChessboardCorners(gray, size, flags)
        if ret:
            print(f"✅ 成功! size={size}, flags={flags}")
            break
    if ret:
        break
else:
    print("❌ 所有尝试都失败")

# 顺便看看图片的亮度分布
print(f"图像灰度均值: {gray.mean():.1f} (0=全黑, 255=全白)")
print(f"图像灰度最小值: {gray.min()}, 最大值: {gray.max()}")