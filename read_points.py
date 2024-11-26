import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# JSONファイルの読み込み
with open("segment_coordinates.json", "r") as file:
    data = json.load(file)

# 保存用ディレクトリを作成
output_dir = "output_segments"
os.makedirs(output_dir, exist_ok=True)

# セグメントごとの画像とグラフを表示・保存
for segment, info in data.items():
    print(f"\n=== {segment} ===")  # セグメントの識別

    # セグメント画像の読み込み
    image_file = f"{segment}.jpg"
    image = cv2.imread(image_file)

    if image is None:
        print(f"{image_file} が見つかりません。スキップします。")
        continue

    # プロット準備
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Segment: {segment}")

    # 画像の表示 (左側)
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Segment Image")
    axes[0].axis("off")

    # グラフのプロット (右側)
    axes[1].set_title("Segment Coordinates")
    axes[1].set_xlabel("X Coordinate")
    axes[1].set_ylabel("Y Coordinate")

    for color, points in info["color_points"].items():
        # 色を指定 (例: 青, 緑, 赤, 黄)
        if "color_0" in color:
            color_rgb = "blue"  # 青
        elif "color_1" in color:
            color_rgb = "green"  # 緑
        elif "color_2" in color:
            color_rgb = "red"  # 赤
        elif "color_3" in color:
            color_rgb = "yellow"  # 黄
        else:
            color_rgb = "black"  # その他の色

        # 点群をプロット
        points = np.array(points)
        if points.size > 0:
            axes[1].scatter(points[:, 0], points[:, 1], label=color, c=color_rgb, s=10)

    # グラフ設定
    axes[1].legend()
    axes[1].invert_yaxis()  # OpenCV座標系に合わせてY軸を反転
    axes[1].grid(True)

    # グラフと画像を保存
    combined_file = os.path.join(output_dir, f"{segment}_combined.png")
    plt.savefig(combined_file)
    print(f"画像とグラフを保存しました: {combined_file}")

    # 表示
    plt.show()
