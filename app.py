import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image, ImageOps
import urllib.request
import cv2

st.title("物体検出カメラ（修正版）")

# 1. AIモデルの準備（キャッシュして高速化）
model_url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
model_path = "model.tflite"

@st.cache_resource
def load_model():
    urllib.request.urlretrieve(model_url, model_path)
    return model_path

model_file = load_model()

# 2. カメラ入力
img_file = st.camera_input("写真を撮る")

if img_file is not None:
    # --- 修正ポイント1: 画像の向きを正す ---
    image = Image.open(img_file)
    image = ImageOps.exif_transpose(image) # スマホ特有の回転情報を補正
    
    image_np = np.array(image)
    
    # --- 修正ポイント2: 色の並びを変換 (RGB -> BGR) ---
    # OpenCVはBGRで描画するため、一度変換してコピーを作成
    output_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # MediaPipe用の画像を作成
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

    # 3. オブジェクト検出の設定
    options = vision.ObjectDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=model_file),
        score_threshold=0.3, # 閾値を少し下げて検知しやすくします (0.5 -> 0.3)
    )

    # 4. 検出実行
    with vision.ObjectDetector.create_from_options(options) as detector:
        detection_result = detector.detect(mp_image)

        # 5. 結果の描画
        if detection_result.detections:
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                # 座標計算
                start_point = (int(bbox.origin_x), int(bbox.origin_y))
                end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))

                # 枠を描く (BGRなので (0, 255, 0) は緑)
                cv2.rectangle(output_image, start_point, end_point, (0, 255, 0), 5)

                # ラベル
                category = detection.categories[0]
                label_text = f"{category.category_name}: {int(category.score*100)}%"
                cv2.putText(output_image, label_text, (start_point[0], start_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # --- 修正ポイント3: 表示用にRGBに戻す ---
            final_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            st.image(final_image, caption="検出成功！", use_container_width=True)
            st.success(f"{len(detection_result.detections)} 個見つかりました")
        else:
            # 失敗した場合は元の画像を表示
            st.image(image_np, caption="何も見つかりませんでした")
            st.warning("検知されませんでした。もっと明るい場所で、対象を中央に映してみてください。")
