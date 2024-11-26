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
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # 輪郭検出
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_display = image.copy()

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # 面積フィルタ
            cv2.drawContours(image_display, [cnt], -1, (0, 255, 0), 2)  # 緑の輪郭

            # 多角形近似で4つの点を強制的に検出
            epsilon = 0.05 * cv2.arcLength(cnt, True)  # 近似精度を調整
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # 必要に応じて頂点数を制限
            while len(approx) > 4:
                epsilon *= 1.1  # 近似精度を緩める
                approx = cv2.approxPolyDP(cnt, epsilon, True)

            # 頂点を赤い点で描画
            for point in approx:
                x, y = point[0]
                cv2.circle(image_display, (x, y), 5, (0, 0, 255), -1)  # 赤い点

    # エッジと検出結果を更新
    cv2.imshow('Edge Detection', image_display)
    cv2.imshow('Edges', edges)

def save_cropped_regions_with_contours():
    """緑の枠線と赤点を含む領域を切り出して保存"""
    global contours, image

    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > 500:  # 面積フィルタ
            # 元画像に枠線と赤点を描画
            image_with_annotations = image.copy()
            cv2.drawContours(image_with_annotations, [cnt], -1, (0, 255, 0), 2)  # 緑の枠線

            # 外接矩形で切り出す
            x, y, w, h = cv2.boundingRect(cnt)  # 外接矩形を取得

            # 多角形近似で頂点を取得し赤点を描画
            epsilon = 0.05 * cv2.arcLength(cnt, True)  # 近似精度を調整
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            while len(approx) > 4:
                epsilon *= 1.1
                approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            for point in approx:
                px, py = point[0]
                cv2.circle(image_with_annotations, (px, py), 5, (0, 0, 255), -1)  # 赤い点

            # 切り出し領域に赤点と枠線を含める
            cropped_with_annotations = image_with_annotations[y:y+h, x:x+w]

            # 保存
            filename = f"cropped_region_with_annotations_{i+1}.jpg"
            cv2.imwrite(filename, cropped_with_annotations)  # ファイルに保存
            print(f"枠線と赤点を含む切り出した領域を保存しました: {filename}")

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

    # エッジ検出の初期値
    threshold1_initial = 100
    threshold2_initial = 200
    edges = cv2.Canny(blurred, threshold1_initial, threshold2_initial)

    # エッジ膨張用カーネル
    kernel = np.ones((3, 3), np.uint8)

    # ウィンドウ作成
    cv2.namedWindow('Edge Detection')
    cv2.namedWindow('Edges')

    # トラックバー作成
    cv2.createTrackbar('Threshold1', 'Edge Detection', threshold1_initial, 500, update_edges)
    cv2.createTrackbar('Threshold2', 'Edge Detection', threshold2_initial, 500, update_edges)

    # 初期表示
    contours = []
    update_edges(0)

    print("エッジ検出の閾値を調整してください")
    print("スペースキーを押すと枠線と赤点を含む緑の枠線内を切り出して保存します")

    while True:
        key = cv2.waitKey(1)
        if key == 27:  # Escキーで終了
            break
        elif key == 32:  # スペースキーで切り出しを実行
            save_cropped_regions_with_contours()

    cv2.destroyAllWindows()
    print("スクリプトの実行が完了しました")

except Exception as e:
    print(f"エラーが発生しました: {e}")
