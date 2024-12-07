import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def load_all_points_from_json(json_file_path):
    """
    JSONファイル内の全てのセグメントと対応するcolor_pointsを読み込み、
    { (segment_name, color_name): points_list, ... } という辞書を返す。
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    all_points = {}
    for segment_name, segment_data in data.items():
        color_points = segment_data.get("color_points", {})
        for color_name, points in color_points.items():
            all_points[(segment_name, color_name)] = points

    return all_points

def calculate_wasserstein_distance(points1, points2):
    """
    2つの点群間のWasserstein距離を計算する関数。
    両点群の点数が違う場合は、最小サイズに合わせる簡易的処理。
    """
    min_size = min(len(points1), len(points2))
    points1 = np.array(points1[:min_size])
    points2 = np.array(points2[:min_size])

    # ユークリッド距離行列を計算
    distance_matrix = cdist(points1, points2, metric='euclidean')

    # ハンガリー法で最適割り当て
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # 割り当てたペア間距離合計
    wasserstein_distance = distance_matrix[row_ind, col_ind].sum()
    return wasserstein_distance

def plot_points(points1, points2):
    """
    2つの点群をプロットする関数。
    """
    points1 = np.array(points1)
    points2 = np.array(points2)

    plt.figure(figsize=(8, 6))
    plt.scatter(points1[:, 0], points1[:, 1], label="Reference (segment_1, color_3)", c='blue', marker='o')
    plt.scatter(points2[:, 0], points2[:, 1], label="Comparison Points", c='red', marker='x')
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.title("Visualization of Two Point Sets")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # JSONファイルのパス
    json_file_path = "segment_coordinates.json"
    
    # 全ての点群を取得
    all_points_dict = load_all_points_from_json(json_file_path)

    # 基準点群：segment_1 の color_3
    reference_key = ("segment_1", "color_0")
    if reference_key not in all_points_dict:
        raise ValueError("The reference segment_1 color_3 does not exist in the JSON.")

    reference_points = all_points_dict[reference_key]
    reference_count = len(reference_points)

    # 比較対象とするcolorを0～3に限定
    valid_colors = {"color_0", "color_1", "color_2", "color_3"}

    # 基準以外の全ての点群との距離を計算（color_0～color_3 のみ）
    # さらに、点数が±10以上異なる場合は比較しない
    distances = []
    for (seg_name, col_name), points in all_points_dict.items():
        if (seg_name, col_name) == reference_key:
            continue  # 基準と同じ点群はスキップ
        if col_name in valid_colors:
            # 点数差チェック
            if abs(len(points) - reference_count) <= 10:
                dist = calculate_wasserstein_distance(reference_points, points)
                distances.append(((seg_name, col_name), dist))
            else:
                # 点数差が±10を超える場合はスキップ
                pass

    # 距離でソート（距離が小さい＝類似度が高い）
    distances.sort(key=lambda x: x[1])

    # 上位３つを取得
    top_3 = distances[:3]

    # 結果を表示
    print("Top 3 similar point sets to segment_1 color_3 (considering only color_0 to color_3 and point-count difference <= 10):")
    for rank, ((seg_name, col_name), dist) in enumerate(top_3, start=1):
        print(f"{rank}. {seg_name}, {col_name}: 距離={dist}")

    # 参考までに1つ目の類似点群との比較をプロット
    if top_3:
        best_match_key, best_match_distance = top_3[0]
        best_points = all_points_dict[best_match_key]
        print(f"\n最も類似度が高い(距離最小)の点群 ({best_match_key[0]}, {best_match_key[1]}) をプロットします。")
        plot_points(reference_points, best_points)
