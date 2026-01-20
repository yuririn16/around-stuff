import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image

# 1. オブジェクト検出の準備
mp_object_detection = mp.solutions.object_detection
mp_drawing = mp.solutions.drawing_utils

st.title("物体検出（Object Detection）練習中")
st.write("身の回りのものを撮ってみよう（スマホ、キーボード、コップなど）")

img_file = st.camera_input("写真を撮る")

if img_file is not None:
    image = Image.open(img_file)
    image_np = np.array(image)

    # 2. オブジェクト検出AIを起動
    # model_selection=0 は近距離（2m以内）、1は遠距離（5m以内）
    with mp_object_detection.ObjectDetection(model_selection=0, min_detection_confidence=0.5) as object_detection:
        results = object_detection.process(image_np)

        # 3. 見つけた物体に枠と名前を描く
        if results.detections:
            for detection in results.detections:
                # 枠を描く
                mp_drawing.draw_detection(image_np, detection)
                
                # 物体の名前（ラベル）を取得して表示
                label = detection.label_id[0] # 本来はIDから名前を引きますが、練習用にIDを表示
                st.write(f"何かを見つけました！ (検出スコア: {int(detection.score[0]*100)}%)")

            st.image(image_np, caption="検出結果", use_container_width=True)
        else:
            st.warning("何も見つかりませんでした。もっと近づけてみてください。")
