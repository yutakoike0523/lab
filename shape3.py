# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像の読み込み
image_path = 'cropped_region_with_contours_3.jpg'  # 必要に応じて画像パスを修正
image = cv2.imread(image_path)

# エラー処理: 画像が読み込めない場合
if image is None:
    raise FileNotFoundError("画像が読み込めませんでした。パスを確認してください。")

# HSV色空間に変換
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 緑色の範囲を定義 (調整可能)
lower_green = np.array([40, 100, 100])  # 下限値
upper_green = np.array([80, 255, 255])  # 上限値

# 緑色部分をマスクで抽出
mask = cv2.inRange(hsv, lower_green, upper_green)
masked_image = cv2.bitwise_and(image, image, mask=mask)

# 輪郭を検出
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 輪郭の中で最大のものを選択
if contours:
    largest_contour = max(contours, key=cv2.contourArea)

    # 輪郭点を抽出
    points = largest_contour[:, 0, :]  # (x, y) 座標の配列
    x = points[:, 0]  # x座標
    y = points[:, 1]  # y座標

    # 曲線を3次多項式で近似
    poly_coeffs = np.polyfit(x, y, 3)
    poly_func = np.poly1d(poly_coeffs)

    # 曲線をプロット用に生成
    x_plot = np.linspace(x.min(), x.max(), 500)
    y_plot = poly_func(x_plot)

    # 結果を画像上に描画
    result_image = image.copy()
    for i in range(len(x_plot) - 1):
        pt1 = (int(x_plot[i]), int(y_plot[i]))
        pt2 = (int(x_plot[i + 1]), int(y_plot[i + 1]))
        cv2.line(result_image, pt1, pt2, (0, 0, 255), 2)  # 赤線でプロット

    # プロットと式の出力
    print(f"近似式: y = {poly_func}")
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title("Green Contour with Fitted Curve")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

else:
    print("緑色の輪郭が検出されませんでした。")
