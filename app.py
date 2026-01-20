import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image
import urllib.request

st.title("物体検出（最新版API）練習中")

# 1. AIモデル（学習済みデータ）をダウンロードする設定
# 初回起動時にGoogleのサーバーからモデルを読み込みます
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
    image = Image.open(img_file)
    image_np = np.array(image)
    
    # MediaPipe用の画像形式に変換
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

    # 3. オブジェクト検出の設定
    options = vision.ObjectDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=model_file),
        score_threshold=0.5, # 50%以上の自信があるものだけ表示
    )

    # 4. 検出実行
    with vision.ObjectDetector.create_from_options(options) as detector:
        detection_result = detector.detect(mp_image)

        # 5. 結果の表示
        if detection_result.detections:
            for detection in detection_result.detections:
                # 見つけた物の名前を取得
                category = detection.categories[0]
                label = category.category_name
                score = category.score
                
                st.write(f"✅ **{label}** を発見！ (確信度: {int(score*100)}%)")
            
            st.image(image_np, caption="撮影した画像")
        else:
            st.warning("何も見つかりませんでした。")
