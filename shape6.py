# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(image):
    """画像を処理してセグメントごとに曲線式を計算し、プロット"""
    # グレースケール変換とぼかし
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # エッジ検出と膨張
    edges = cv2.Canny(blurred, 100, 200)
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # 輪郭検出
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_display = image.copy()

    for idx, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > 500:  # 面積フィルタ
            draw_colored_segments(image_display, cnt)

    # 処理結果を表示
    cv2.imshow("Processed Image", image_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_colored_segments(image, contour):
    """元の緑の枠線に沿って4色に分割して描画し、曲線式を計算してプロット"""
    # 多角形近似で4つの点を検出
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # 必要に応じて頂点数を制限
    while len(approx) > 4:
        epsilon *= 1.1  # 近似精度を緩める
        approx = cv2.approxPolyDP(contour, epsilon, True)

    # 頂点の座標を取得
    points = [tuple(pt[0]) for pt in approx]

    # 色リスト
    colors = ['blue', 'green', 'red', 'cyan']  # セグメントの色リスト

    # 各セグメントごとに処理
    for i in range(len(points)):
        start_point = points[i]
        end_point = points[(i + 1) % len(points)]

        # セグメント内の輪郭点を取得
        segment_points = get_segment_points(contour, start_point, end_point)

        # 曲線式を計算してプロット
        if len(segment_points) > 1:
            fit_curve_and_plot(segment_points, colors[i % len(colors)], i + 1)

        # 曲線を画像に描画
        for j in range(len(segment_points) - 1):
            pt1 = tuple(segment_points[j])
            pt2 = tuple(segment_points[j + 1])
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)  # 緑色で描画

def get_segment_points(contour, start_point, end_point):
    """セグメント内の輪郭点を取得"""
    start_idx = -1
    end_idx = -1

    # 輪郭の中でstart_pointとend_pointのインデックスを探す
    for idx, pt in enumerate(contour[:, 0, :]):
        if tuple(pt) == start_point:
            start_idx = idx
        if tuple(pt) == end_point:
            end_idx = idx

    # インデックス範囲に基づいてセグメントのポイントを取得
    if start_idx < end_idx:
        return contour[start_idx:end_idx + 1, 0, :]
    else:  # 輪郭が閉じている場合の対応
        return np.vstack((contour[start_idx:, 0, :], contour[:end_idx + 1, 0, :]))

def fit_curve_and_plot(segment_points, color, segment_id):
    """セグメント点に基づいて曲線を近似し、グラフにプロット"""
    x = segment_points[:, 0]
    y = segment_points[:, 1]

    # 7次多項式で近似
    coefficients = np.polyfit(x, y, 7)
    polynomial = np.poly1d(coefficients)

    # コンソールに曲線式を出力
    print(f"セグメント {segment_id} の曲線式: {polynomial}")

    # 曲線をプロット
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, color=color, label=f'Segment {segment_id} Points')
    x_curve = np.linspace(min(x), max(x), 100)
    y_curve = polynomial(x_curve)
    plt.plot(x_curve, y_curve, color='black', label=f'Segment {segment_id} Curve')
    plt.title(f"Segment {segment_id} Curve (7th degree)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

try:
    print("スクリプトが開始されました")

    # 画像の読み込み（ユーザーがアップロードした画像を指定）
    image = cv2.imread('segment_1.jpg')  # 適切なパスに変更してください
    if image is None:
        raise FileNotFoundError("指定された画像が見つかりません。パスを確認してください。")

    print("画像の読み込みが成功しました")

    # 画像を処理
    process_image(image)

    print("スクリプトの実行が完了しました")

except Exception as e:
    print(f"エラーが発生しました: {e}")
