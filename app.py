import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import urllib.request
import os

# 日本語ラベル辞書
LABEL_MAP = {
    "person": "人間", "bicycle": "自転車", "car": "車", "motorcycle": "バイク",
    "bottle": "ボトル", "cup": "コップ", "chair": "椅子", "tv": "テレビ",
    "laptop": "PC", "mouse": "マウス", "keyboard": "キーボード", "cell phone": "スマホ"
}

st.set_page_config(page_title="AIカメラ", layout="centered")
st.title("🎨 カラー別・AI物体検出")

# 1. モデル準備
model_path = "model.tflite"
model_url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"

@st.cache_resource
def load_model_file():
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(model_url, model_path)
    return model_path

try:
    m_file = load_model_file()
except:
    st.error("モデルの読み込みに失敗しました")

# 2. カメラ入力
img_file = st.camera_input("撮影する")

if img_file is not None:
    # 画像準備
    img = Image.open(img_file)
    img = ImageOps.exif_transpose(img)
    img_np = np.array(img).astype(np.uint8)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)

    # 3. AIの設定 (行を短く分割してエラー防止)
    base_ops = python.BaseOptions(model_asset_path=m_file)
    options = vision.ObjectDetectorOptions(
        base_options=base_ops,
        score_threshold=0.3
    )

    # 4. 実行と表示
    try:
        with vision.ObjectDetector.create_from_options(options) as detector:
            res = detector.detect(mp_img)
            draw_img = img.copy()
            draw = ImageDraw.Draw(draw_img)
            
            # 鮮やかな色のリスト
            COLORS = ["#FF3B30", "#4CD964", "#007AFF", "#FFCC00", "#FF9500", "#5856D6"]

            if res.detections:
                for i, det in enumerate(res.detections):
                    # 色の選択
                    c_color = COLORS[i % len(COLORS)]
                    
                    # 座標
                    b = det.bounding_box
                    rect = [b.origin_x, b.origin_y, b.origin_x + b.width, b.origin_y + b.height]
                    
                    # 描画 (枠)
                    draw.rectangle(rect, outline=c_color, width=8)
                    
                    # ラベル
                    cat = det.categories[0]
                    name = LABEL_MAP.get(cat.category_name, cat.category_name)
                    txt = f"{name} {int(cat.score*100)}%"
                    
                    # ラベル背景
                    draw.rectangle([rect[0], rect[1]-35, rect[0]+len(txt)*18, rect[1]], fill=c_color)
                    draw.text((rect[0]+5, rect[1]-30), txt, fill="white")
                
                st.image(draw_img, use_container_width=True)
                
                # レポート
                st.subheader("📊 検出結果")
                for i, det in enumerate(res.detections):
                    cat = det.categories[0]
                    n = LABEL_MAP.get(cat.category_name, cat.category_name)
                    st.markdown(f"<span style='color:{COLORS[i%len(COLORS)]}'>●</span> {n}", unsafe_allow_html=True)
                    st.progress(float(cat.score))
            else:
                st.image(img, use_container_width=True)
                st.info("何も検知されませんでした")
    except Exception as e:
        st.error(f"実行エラー: {e}")
