# -*- coding: utf-8 -*-
import cv2
import numpy as np

def split_and_draw_contour(image, contour):
    """輪郭を上下左右に分割して色分け"""
    # 輪郭点を取得
    points = contour[:, 0, :]  # (x, y)形式
    x = points[:, 0]  # x座標
    y = points[:, 1]  # y座標

    # 上下左右の分割基準
    x_center = (x.min() + x.max()) // 2
    y_center = (y.min() + y.max()) // 2

    # 分割結果を保持
    parts = {"top": [], "bottom": [], "left": [], "right": []}

    for px, py in zip(x, y):
        if py <= y.min() + 10:  # 上の部分
            parts["top"].append((px, py))
        elif py >= y.max() - 10:  # 下の部分
            parts["bottom"].append((px, py))
        elif px <= x.min() + 10:  # 左の部分
            parts["left"].append((px, py))
        elif px >= x.max() - 10:  # 右の部分
            parts["right"].append((px, py))

    # 元画像をコピーして描画
    color_image = image.copy()
    colors = {
        "top": (255, 0, 0),    # 青
        "bottom": (0, 255, 0), # 緑
        "left": (0, 0, 255),   # 赤
        "right": (255, 255, 0) # 水色
    }

    # 分割した各部分を描画
    for part, points in parts.items():
        for px, py in points:
            cv2.circle(color_image, (px, py), 2, colors[part], -1)

    return color_image, parts

try:
    print("スクリプトが開始されました")

    # 画像の読み込み
    image_path = 'cropped_region_with_contours_1.jpg'  # 必要に応じて画像パスを変更
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("指定された画像が見つかりません。パスを確認してください。")

    print("画像の読み込みが成功しました")

    # 緑色部分を検出するためにHSV変換
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 100, 100])  # 緑色の下限値
    upper_green = np.array([80, 255, 255])  # 緑色の上限値
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 輪郭検出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 最大の輪郭を取得
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # 輪郭を上下左右に分割して色分け
        color_image, parts = split_and_draw_contour(image, largest_contour)

        # 結果を表示
        cv2.imshow("Colored Contour", color_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("輪郭を上下左右に分割しました。")
        print(f"各部分の点数: 上({len(parts['top'])}), 下({len(parts['bottom'])}), 左({len(parts['left'])}), 右({len(parts['right'])})")
    else:
        print("緑の輪郭が検出されませんでした。")

except Exception as e:
    print(f"エラーが発生しました: {e}")
