import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image, ImageOps
import urllib.request
import cv2
import os

st.title("物体検出カメラ（リスト表示版）")

# 1. モデル準備
model_path = "model.tflite"
model_url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"

@st.cache_resource
def load_model_file():
    if not os.path.exists(model_path):
        with st.spinner("AIモデルを準備中..."):
            urllib.request.urlretrieve(model_url, model_path)
    return model_path

try:
    model_file = load_model_file()
except Exception as e:
    st.error(f"モデルの準備エラー: {e}")

# 2. カメラ入力
img_file = st.camera_input("写真を撮る")

if img_file is not None:
    image = Image.open(img_file)
    image = ImageOps.exif_transpose(image)
    image_np = np.array(image).astype(np.uint8)
    
    # 描画用画像の作成
    output_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

    # 3. 検出器の設定
    base_options = python.BaseOptions(model_asset_path=model_file)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        score_threshold=0.2,
        max_results=10
    )

    # 4. 実行と表示
    try:
        with vision.ObjectDetector.create_from_options(options) as detector:
            result = detector.detect(mp_image)

            if result.detections:
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]
                # リスト用のデータを溜める変数
                found_items = []

                for i, detection in enumerate(result.detections):
                    # 枠とラベルの描画
                    color = colors[i % len(colors)]
                    box = detection.bounding_box
                    x, y, w, h = int(box.origin_x), int(box.origin_y), int(box.width), int(box.height)
                    cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 3)
                    
                    cat = detection.categories[0]
                    label = f"{cat.category_name} {int(cat.score*100)}%"
                    cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # --- リスト表示用にデータを追加 ---
                    found_items.append({
                        "name": cat.category_name,
                        "score": int(cat.score * 100)
                    })

                # 加工した画像の表示
                final_img = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                st.image(final_img, caption="検出成功", use_container_width=True)

                # --- 画面下部のリスト表示 ---
                st.write("---")
                st.subheader("📋 検出された物体のリスト")
                for item in found_items:
                    st.write(f"✅ **{item['name']}** (確信度: {item['score']}%)")
                
                st.success(f"合計 {len(found_items)} 個の物体が見つかりました")

            else:
                st.image(image_np, caption="何も見つかりませんでした")
                st.info("明るい場所で撮り直してみてください。")

    except Exception as e:
        st.error(f"AI実行エラー: {e}")
