import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from itertools import combinations
import json

def polynomial_fit(x, a, b, c):
    """2次多項式モデル"""
    return a * x**2 + b * x + c


def fit_curve(points):
    """点群に対して曲線をフィッティング"""
    x = points[:, 0]
    y = points[:, 1]
    params, _ = curve_fit(polynomial_fit, x, y)
    return params


def calculate_similarity(params1, params2):
    """2つの曲線の類似度を計算 (平均二乗誤差: MSE)"""
    return np.mean((np.array(params1) - np.array(params2))**2)


def process_segments(segment_data):
    """セグメントごとの曲線をフィッティングし、類似度を計算"""
    curve_params = {}  # セグメントごとのフィッティングパラメータを保存
    similarities = []  # 類似度スコアを保存

    for segment_id, data in segment_data.items():
        curve_params[segment_id] = {}
        for color, points in data["color_points"].items():
            points = np.array(points)
            if points.shape[0] > 2:  # 点が十分に多い場合のみフィッティング
                params = fit_curve(points)
                curve_params[segment_id][color] = params

    # セグメント間の類似度を計算
    for (seg1, data1), (seg2, data2) in combinations(curve_params.items(), 2):
        for color1, params1 in data1.items():
            for color2, params2 in data2.items():
                similarity = calculate_similarity(params1, params2)
                similarities.append(((seg1, color1), (seg2, color2), similarity))

    # 最も類似度が高いペアを選択
    most_similar_pair = min(similarities, key=lambda x: x[2])  # 類似度スコアが最小のペアを選択
    return most_similar_pair, curve_params


# サンプルデータの読み込み
with open("segment_coordinates.json", "r") as json_file:
    segment_data = json.load(json_file)

# 曲線フィッティングと類似度計算
most_similar_pair, curve_params = process_segments(segment_data)

# 結果の表示
seg1, color1 = most_similar_pair[0]
seg2, color2 = most_similar_pair[1]
similarity_score = most_similar_pair[2]

print(f"最も類似しているペア:")
print(f"セグメント {seg1} の {color1} と セグメント {seg2} の {color2}")
print(f"類似度スコア (MSE): {similarity_score}")

# フィッティング結果を可視化
for segment_id, data in curve_params.items():
    plt.figure(figsize=(8, 6))
    plt.title(f"Segment: {segment_id}")
    for color, params in data.items():
        # segment_id に "segment_" が既に含まれているか確認
        segment_key = segment_id if segment_id.startswith("segment_") else f"segment_{segment_id}"
        aaaaaaaaaaaaaaaapoints = np.array(segment_data[segment_key]["color_points"][color])

        x = points[:, 0]
        y = points[:, 1]
        plt.scatter(x, y, label=color)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = polynomial_fit(x_fit, *params)
        plt.plot(x_fit, y_fit, label=f"{color} Fit", linestyle="--")
    plt.legend()
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.gca().invert_yaxis()
    plt.grid()
    plt.show()
