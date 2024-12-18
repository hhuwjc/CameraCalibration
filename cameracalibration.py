import cv2
import numpy as np
import math
import glob

# ----------- 内参标定 -----------
# 棋盘格规格（行内角点数 x 列内角点数）
chessboard_size = (12, 8)

objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# 存储棋盘格的世界坐标和图像坐标
objpoints = []  # 世界坐标
imgpoints = []  # 图像坐标

# 加载标定图片
images = glob.glob('calibration_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"无法加载图片: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        print(f"成功检测到棋盘格角点: {fname}")
        objpoints.append(objp)
        imgpoints.append(corners)

        # 可视化检测到的角点
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(500)
    else:
        print(f"未检测到棋盘格角点: {fname}")

cv2.destroyAllWindows()

# 验证标定数据是否为空
if len(objpoints) == 0 or len(imgpoints) == 0:
    print("未找到足够的标定数据，请检查标定图片是否有效或棋盘格角点是否清晰。")
    exit()

# 调用 cv2.calibrateCamera 进行标定
ret, camera_matrix, distortion_coeff, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# 打印标定结果
print("相机内参矩阵 (Camera Matrix):")
print(camera_matrix)

print("\n畸变系数 (Distortion Coefficients):")
print(distortion_coeff.ravel())

# 保存标定结果到文件
np.savetxt("camera_matrix.txt", camera_matrix, fmt='%f')
np.savetxt("distortion_coeff.txt", distortion_coeff, fmt='%f')

# ----------- 图像畸变校正 -----------
# 读取需要校正的图像
undistort_image_path = 'undistort_input.jpg'  # 替换为你的图片路径
input_image = cv2.imread(undistort_image_path)

if input_image is not None:
    # 获取图像尺寸
    h, w = input_image.shape[:2]

    # 计算优化后的内参矩阵（自由比例 alpha = 0）
    optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, distortion_coeff, (w, h), 0, (w, h)
    )

    # 进行畸变校正
    undistorted_image = cv2.undistort(input_image, camera_matrix, distortion_coeff, None, optimal_camera_matrix)

    # 裁剪校正后的图像（根据 ROI）
    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y+h, x:x+w]

    # 显示和保存校正后的图像
    cv2.imshow("Original Image", input_image)
    cv2.imshow("Undistorted Image", undistorted_image)
    cv2.imwrite("undistorted_output.jpg", undistorted_image)  # 保存校正后的图像
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"无法找到图像: {undistort_image_path}")

# ----------- 外参计算（俯仰角计算） -----------
try:
    # 提示用户输入变量值
    Y_A = float(input("请输入A点纵坐标 Y_A (单位：米): "))
    y_b = float(input("请输入B点像素纵坐标 y_b (单位：像素): "))
    y_a = float(input("请输入A点像素纵坐标 y_a (单位：像素): "))
    h1 = float(input("AB两点间的距离 h1 (单位：米): "))
    beta_degrees = float(input("平行线与世界坐标系夹角 β (单位：度): "))
    x_e = float(input("请输E点像素横坐标 x_e (单位：像素): "))
    x_a = float(input("请输A点像素横坐标 x_a (单位：像素): "))
    d = float(input("两平行线间的距离 d (单位：米): "))

    # 将 β 转换为弧度
    beta = math.radians(beta_degrees)

    # 根据公式计算俯仰角 theta
    numerator = d * (Y_A * y_b - Y_A * y_a + h1 * y_b * math.sin(beta))
    denominator = h1 * Y_A * (x_e - x_a) * (math.sin(beta) ** 2)

    # 防止输入无效导致错误
    if denominator != 0 and -1 <= numerator / denominator <= 1:
        theta = math.asin(numerator / denominator)  # 计算俯仰角 (弧度)
        theta_degrees = math.degrees(theta)         # 转换为角度
        print(f"俯仰角 (弧度): {theta:.4f}")
        print(f"俯仰角 (角度): {theta_degrees:.2f}°")
    else:
        print("输入参数无效，无法计算俯仰角，请检查输入值。")
except ValueError:
    print("输入无效，请输入数字。")



