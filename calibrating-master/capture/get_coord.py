import cv2

img_path = r"E:\learning\calibrating\calibrating-master\capture\object_images\2\stereo_l.jpg"
img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"读不到图片: {img_path}")

# 缩小显示，避免超出屏幕
scale = 0.6
disp = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
window = "Click on the object center (q to quit)"

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 还原到原图坐标
        ox = int(x / scale)
        oy = int(y / scale)
        print(f"原图坐标: ({ox}, {oy})")
        marked = img.copy()
        cv2.drawMarker(marked, (ox, oy), (0, 255, 255), cv2.MARKER_CROSS, 30, 3)
        cv2.imshow(window, cv2.resize(marked, (int(marked.shape[1]*scale), int(marked.shape[0]*scale))))

cv2.namedWindow(window)
cv2.setMouseCallback(window, on_mouse)
cv2.imshow(window, disp)
print("点击手柄中心，终端会打印坐标。按 q 退出。")

while True:
    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()