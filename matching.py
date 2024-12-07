import json
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

# segment_coordinates.json を読み込む
with open("segment_coordinates.json", "r") as file:
    segment_coordinates = json.load(file)

def calculate_similarity(points1, points2):
    """
    点群同士の類似度を計算する（平均最近傍距離を利用）
    """
    tree1 = KDTree(points1)
    tree2 = KDTree(points2)
    
    # points1 -> points2 の最近傍距離
    distances1, _ = tree1.query(points2)
    avg_distance1 = np.mean(distances1)
    
    # points2 -> points1 の最近傍距離
    distances2, _ = tree2.query(points1)
    avg_distance2 = np.mean(distances2)
    
    # 平均距離を返す（双方向）
    return (avg_distance1 + avg_distance2) / 2

# すべてのセグメントと色を取得
segments = list(segment_coordinates.keys())
colors = list(segment_coordinates[segments[0]]['color_points'].keys())

# 結果を格納するリスト
results = []

# 凸（例: Segment 1, color_0）の点群を基準に比較
reference_segment = "segment_1"
reference_color = "color_0"

# 基準点群の存在を確認
if reference_segment not in segment_coordinates or reference_color not in segment_coordinates[reference_segment]['color_points']:
    print(f"Error: {reference_segment} {reference_color} does not exist in the data.")
    exit()

reference_points = np.array(segment_coordinates[reference_segment]['color_points'][reference_color])

for segment in segments:
    for color in colors:
        # セグメントや色が存在するかを確認
        if 'color_points' not in segment_coordinates[segment] or color not in segment_coordinates[segment]['color_points']:
            print(f"Warning: {segment} {color} does not exist.")
            continue
        
        # 同じセグメントと色はスキップ
        if segment == reference_segment and color == reference_color:
            continue
        
        comparison_points = np.array(segment_coordinates[segment]['color_points'][color])
        similarity = calculate_similarity(reference_points, comparison_points)
        
        # 結果を保存
        results.append((segment, color, similarity))

# 類似度でソート（小さいほど類似）
results = sorted(results, key=lambda x: x[2])

# トップ5を表示
print("Top 5 most similar lines to", reference_segment, reference_color)
for i, (segment, color, similarity) in enumerate(results[:5], 1):
    print(f"{i}. Segment: {segment}, Color: {color}, Similarity: {similarity:.2f}")

# 可視化（参考）
plt.figure(figsize=(12, 8))
plt.title("Top 5 Most Similar Lines Visualization")

# 基準点群をプロット
plt.scatter(reference_points[:, 0], reference_points[:, 1], c='cyan', label=f"Reference {reference_segment} {reference_color}")

# トップ5の類似点群をプロット
for i, (segment, color, _) in enumerate(results[:5], 1):
    comparison_points = np.array(segment_coordinates[segment]['color_points'][color])
    plt.scatter(comparison_points[:, 0], comparison_points[:, 1], label=f"Rank {i}: {segment} {color}", alpha=0.6)

# 軸と凡例
plt.gca().invert_yaxis()
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.show()
