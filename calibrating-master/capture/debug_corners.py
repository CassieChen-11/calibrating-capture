import cv2

img_path = r"E:\learning\calibrating\calibrating-master\capture\calibration_images\1\stereo_l.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 亮度增强后再尝试检测
enhanced = cv2.equalizeHist(gray)

sizes = [(7,10),(10,7),(8,11),(11,8),(6,9),(9,6),(7,9),(9,7),(8,10),(10,8)]
print("=== 原图检测 ===")
for size in sizes:
    ret, _ = cv2.findChessboardCorners(gray, size, None)
    if ret:
        print(f"✅ 原图成功: {size}")
        break
else:
    print("原图全部失败")

print("\n=== 直方图均衡化后检测 ===")
for size in sizes:
    ret, _ = cv2.findChessboardCorners(enhanced, size, None)
    if ret:
        print(f"✅ 增强后成功: {size}")
        break
else:
    print("增强后也全部失败")

# 显示增强后的图，方便直观查看
cv2.imshow("Enhanced", enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()