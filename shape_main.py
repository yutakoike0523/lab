# -*- coding: utf-8 -*-
import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt


def update_edges(val):
    """トラックバーの値を使ってエッジ検出を更新"""
    global edges, blurred, image_display, contours

    # トラックバーから閾値を取得
    threshold1 = cv2.getTrackbarPos('Threshold1', 'Edge Detection')
    threshold2 = cv2.getTrackbarPos('Threshold2', 'Edge Detection')

    # エッジ検出を更新
    edges = cv2.Canny(blurred, threshold1, threshold2)
    
    # エッジ膨張用カーネルを定義
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # 輪郭検出
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_display[:] = image[:]  # 元画像をコピー

    for idx, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > 500:  # 面積フィルタ
            draw_colored_segments(image_display, cnt, idx)

    # エッジと検出結果を更新
    cv2.imshow('Edge Detection', image_display)
    cv2.imshow('Edges', edges)

def draw_colored_segments(image, contour, segment_idx):
    """元の枠線に沿って描画し、ピークと谷を検出"""
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # 頂点の座標を取得
    points = [tuple(pt[0]) for pt in approx]

    # セグメントが空の場合はスキップ
    if not points:
        print(f"セグメント {segment_idx} に頂点が存在しません。スキップします。")
        return

    # 色リストを固定（青, 緑, 赤, 水色）
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # 必要な色だけ
    colors = base_colors * ((len(points) // len(base_colors)) + 1)  # 必要な色数を確保
    colors = colors[:len(points)]  # 長さをポイント数に制限

    # セグメントごとのデータを初期化
    if segment_idx not in segment_data:
        segment_data[segment_idx] = {"color_points": {f"color_{i}": [] for i in range(len(colors))}}

    for i in range(len(points)):
        start_point = points[i]
        end_point = points[(i + 1) % len(points)]

        # セグメント内の輪郭点を取得
        segment_points = get_segment_points(contour, start_point, end_point)

        # 曲線を描画 (範囲外アクセスを防ぐ)
        if i < len(colors):  # 範囲外アクセスを防ぐ
            for j in range(len(segment_points) - 1):
                pt1 = tuple(segment_points[j])
                pt2 = tuple(segment_points[j + 1])
                cv2.line(image, pt1, pt2, colors[i], 2)

        # ピークと谷を検出してプロット
        detect_peaks_and_valleys(segment_points, colors[i], segment_idx, i)







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

def detect_peaks_and_valleys(segment_points, color, segment_idx, color_idx):
    """セグメント内のピークと谷を検出してプロット"""
    x = segment_points[:, 0]
    y = segment_points[:, 1]

    # 1次微分と2次微分を計算
    dy = np.gradient(y)
    ddy = np.gradient(dy)

    # ピークと谷を検出
    peaks = np.where((ddy < -0.01))[0]  # ピーク
    valleys = np.where((ddy > 0.01))[0]  # 谷

    # セグメントデータ内のこの色の点群を初期化
    segment_data[segment_idx]["color_points"][f"color_{color_idx}"] = []

    # ピークと谷を描画（重複を防ぐ）
    unique_points = set()  # 一意性を確保するためのセット
    for idx in np.concatenate((peaks, valleys)):  # ピークと谷をまとめて処理
        point = (int(x[idx]), int(y[idx]))
        if point not in unique_points:  # 新しい点だけ追加
            unique_points.add(point)
            segment_data[segment_idx]["color_points"][f"color_{color_idx}"].append(point)
            print(f"セグメント {segment_idx}, 色 {color_idx}, 点: {point}")  # デバッグ出力
            cv2.circle(image_display, point, 5, color, -1)  # 描画



def visualize_and_save_segments():
    """セグメントごとの画像とグラフを同時に表示し、指定したセグメントの点のみを描画して保存する"""
    output_dir = "output_segments"
    os.makedirs(output_dir, exist_ok=True)

    json_data = {}
    for idx, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > 500:
            # 外接矩形で図形を切り出し
            x, y, w, h = cv2.boundingRect(cnt)
            segment = image_display[y:y+h, x:x+w]
            current_segment_data = segment_data.get(idx, {})
            json_data[f"segment_{idx + 1}"] = current_segment_data

            # グラフと画像を並べて可視化
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle(f"Segment: {idx + 1}")

            # セグメント画像を表示 (左側)
            axes[0].imshow(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Segment Image")
            axes[0].axis("off")

            # グラフをプロット (右側)
            axes[1].set_title("Segment Coordinates")
            axes[1].set_xlabel("X Coordinate")
            axes[1].set_ylabel("Y Coordinate")

            # 固定色リスト (4色のみ)
            base_colors = ["blue", "green", "red", "cyan"]
            color_idx = 0

            # 現在のセグメントだけの点をプロット
            for color, points in current_segment_data["color_points"].items():
                if color_idx >= len(base_colors):  # 4色のみ使用
                    break

                color_rgb = base_colors[color_idx]
                points = np.array(points)

                if points.size > 0:
                    axes[1].scatter(points[:, 0], points[:, 1], label=f"color_{color_idx}", c=color_rgb, s=10)
                    color_idx += 1  # 次の色に進む

            # プロット範囲を画像全体に設定
            axes[1].set_xlim(0, image.shape[1])  # X軸を画像の幅に合わせる
            axes[1].set_ylim(image.shape[0], 0)  # Y軸を画像の高さに合わせ、反転

            # 凡例と軸設定
            axes[1].legend()
            axes[1].grid(True)

            # 可視化結果の保存
            combined_file = os.path.join(output_dir, f"segment_{idx + 1}_combined.png")
            plt.savefig(combined_file)
            print(f"グラフと画像を保存しました: {combined_file}")

            # 表示
            plt.show()

    # JSONファイルに点群データを保存
    with open("segment_coordinates.json", "w") as json_file:
        json.dump(json_data, json_file, indent=4)
        print("セグメントの座標データをsegment_coordinates.jsonに保存しました。")




try:
    print("スクリプトが開始されました")

    # 画像の読み込み
    image = cv2.imread('img\\2.jpg')  # 適切なパスに変更してください
    if image is None:
        raise FileNotFoundError("指定された画像が見つかりません。パスを確認してください。")

    print("画像の読み込みが成功しました")

    # グレースケール変換とぼかし
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 初期値の設定
    edges = None
    contours = []
    segment_data = {}  # 各セグメントのデータを格納する辞書
    image_display = image.copy()  # 初期化

    # ウィンドウ作成
    cv2.namedWindow('Edge Detection')
    cv2.namedWindow('Edges')

    # トラックバー作成
    cv2.createTrackbar('Threshold1', 'Edge Detection', 100, 500, update_edges)
    cv2.createTrackbar('Threshold2', 'Edge Detection', 200, 500, update_edges)

    # 初期表示
    update_edges(0)

    print("エッジ検出の閾値を調整してください")
    print("スペースキーを押すと図形（セグメント）ごとの画像を保存し、点群データをJSONに保存します")
    print("ESCキーを押して終了します")

    while True:
        key = cv2.waitKey(1)
        if key == 27:  # ESCキーで終了
            break
        elif key == 32:  # スペースキーでセグメントごとの画像と点群データを保存
            visualize_and_save_segments()
    cv2.destroyAllWindows()
    print("スクリプトの実行が完了しました")

except Exception as e:
    print(f"エラーが発生しました: {e}")