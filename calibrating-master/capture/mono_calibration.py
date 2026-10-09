import calibrating
from glob import glob

# 用之前拍好的左相机棋盘格图
board = calibrating.Chessboard(checkboard=(8, 11), size_mm=5)
cam = calibrating.Cam(glob(r"E:\learning\calibrating\calibrating-master\capture\calibration_images\*\stereo_l.jpg"), board)

print(cam)