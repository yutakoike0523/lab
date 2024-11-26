# -*- coding: utf-8 -*-
import cv2
import numpy as np

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
            draw_colored_segments(image_display, cnt)

    # エッジと検出結果を更新
    cv2.imshow('Edge Detection', image_display)
    cv2.imshow('Edges', edges)

def draw_colored_segments(image, contour):
    """元の緑の枠線に沿って4色に分割して描画"""
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
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # 青, 緑, 赤, 黄

    # 各セグメントごとに元の枠線に沿って描画
    for i in range(len(points)):
        start_point = points[i]
        end_point = points[(i + 1) % len(points)]

        # セグメント内の輪郭点を取得
        segment_points = get_segment_points(contour, start_point, end_point)

        # 曲線を描画
        for j in range(len(segment_points) - 1):
            pt1 = tuple(segment_points[j])
            pt2 = tuple(segment_points[j + 1])
            cv2.line(image, pt1, pt2, colors[i % len(colors)], 2)

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

def save_segments():
    """スペースキーを押すと図形（セグメント）ごとの画像を保存"""
    for idx, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > 500:
            # 外接矩形で図形を切り出し
            x, y, w, h = cv2.boundingRect(cnt)
            segment = image_display[y:y+h, x:x+w]
            filename = f"segment_{idx + 1}.jpg"
            cv2.imwrite(filename, segment)
            print(f"図形 {idx + 1} の画像を保存しました: {filename}")

try:
    print("スクリプトが開始されました")

    # 画像の読み込み
    image = cv2.imread('img\\7.jpg')  # 適切なパスに変更してください
    if image is None:
        raise FileNotFoundError("指定された画像が見つかりません。パスを確認してください。")

    print("画像の読み込みが成功しました")

    # グレースケール変換とぼかし
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 初期値の設定
    edges = None
    contours = []
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
    print("スペースキーを押すと図形（セグメント）ごとの画像を保存します")
    print("ESCキーを押して終了します")

    while True:
        key = cv2.waitKey(1)
        if key == 27:  # ESCキーで終了
            break
        elif key == 32:  # スペースキーでセグメントごとの画像を保存
            save_segments()

    cv2.destroyAllWindows()
    print("スクリプトの実行が完了しました")

except Exception as e:
    print(f"エラーが発生しました: {e}")
