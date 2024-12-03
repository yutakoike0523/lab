import json
import numpy as np
import matplotlib.pyplot as plt

# segment_coordinates.json を読み込む
with open("segment_coordinates.json", "r") as file:
    segment_coordinates = json.load(file)

# 凸（Segment 1 - color_0）の点群
convex_points = np.array(segment_coordinates['segment_1']['color_points']['color_0'])

# 凹（Segment 2 - color_3）の点群
concave_points = np.array(segment_coordinates['segment_2']['color_points']['color_3'])

def align_vertical(points):
    """
    点群を垂直方向に揃える関数（Y座標が最小の点を基準に回転）
    """
    # 支点（Y座標が最小の点）
    base_point = points[np.argmin(points[:, 1])]

    # Y座標が最大の点を取得
    max_y_point = points[np.argmax(points[:, 1])]

    # 現在の角度を計算
    dx_current = max_y_point[0] - base_point[0]
    dy_current = max_y_point[1] - base_point[1]
    current_angle = np.arctan2(dy_current, dx_current)

    # 回転させる目標角度は 90 度（π/2）
    target_angle = np.pi / 2

    # 回転角を計算
    rotation_angle = target_angle - current_angle

    # 回転行列を作成
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle)],
        [np.sin(rotation_angle), np.cos(rotation_angle)]
    ])

    # 点群を回転
    aligned_points = (rotation_matrix @ (points - base_point).T).T + base_point

    return aligned_points, base_point, rotation_angle

def rotate_points(points, angle, base_point):
    """
    点群を指定された角度で基準点を中心に回転
    """
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    rotated_points = (rotation_matrix @ (points - base_point).T).T + base_point
    return rotated_points

# 凸を垂直に揃える
convex_aligned, convex_base, _ = align_vertical(convex_points)

# 凹の基準点を取得
concave_base = concave_points[np.argmin(concave_points[:, 1])]

# 凹を凸と向き合うように調整（反転した角度を適用）
_, _, rotation_angle = align_vertical(convex_points)
concave_aligned = rotate_points(concave_points, -rotation_angle, concave_base)

# 凹を凸の基準点に揃えるために平行移動
x_shift = convex_base[0] - concave_base[0]
y_shift = convex_base[1] - concave_base[1]
concave_aligned[:, 0] += x_shift
concave_aligned[:, 1] += y_shift

# 可視化
plt.figure(figsize=(12, 8))
plt.title("Overlay Alignment of Convex and Concave Points")

# 凸（回転後の点群）をプロット（水色）
plt.scatter(convex_aligned[:, 0], convex_aligned[:, 1], c='cyan', label="Aligned Convex Points (color_0)", alpha=0.8)

# 凹（平行移動後の点群）をプロット（オレンジ色）
plt.scatter(concave_aligned[:, 0], concave_aligned[:, 1], c='orange', label="Aligned Concave Points (color_3)", alpha=0.8)

# 支点をマーク
plt.scatter([convex_base[0]], [convex_base[1]], c='black', marker='x', label="Convex Base Point")
plt.scatter([concave_base[0] + x_shift], [concave_base[1] + y_shift], c='purple', marker='x', label="Concave Base Point (Shifted)")

# 軸の設定
plt.gca().invert_yaxis()
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.show()
