import streamlit as st
from cellpose import models, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure
import io as python_io
from pptx import Presentation
from PIL import Image
import torch

st.set_page_config(page_title="Sphere Analyzer v2.1", layout="wide")

st.title("🔴 スフェア自動計測ツール v2.1")
st.write("JPG/PNGまたはPPTXをアップロードしてください。")

with st.sidebar:
    st.header("1. 倍率設定")
    mag = st.radio("顕微鏡の倍率を選択:", ("4x", "10x", "カスタム"))
    if mag == "4x": um_per_pixel = 3.23
    elif mag == "10x": um_per_pixel = 1.28
    else: um_per_pixel = st.number_input("μm/pixelを手入力", value=1.0)

    st.header("2. 解析パラメータ")
    target_diameter = st.slider("予想直径 (px)", 50, 300, 150)
    circularity_threshold = st.slider("真円度のしきい値", 0.5, 0.95, 0.8)

uploaded_files = st.file_uploader("ファイルをドロップ", type=['jpg', 'png', 'pptx', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    images_to_process = []
    for f in uploaded_files:
        if f.name.lower().endswith('.pptx'):
            prs = Presentation(f)
            for i, slide in enumerate(prs.slides):
                for shape in slide.shapes:
                    if shape.shape_type == 13:
                        img_stream = python_io.BytesIO(shape.image.blob)
                        img_np = np.array(Image.open(img_stream))
                        images_to_process.append((f"{f.name}_slide{i+1}", img_np))
        else:
            img_np = np.array(Image.open(f))
            images_to_process.append((f.name, img_np))

    if images_to_process:
        all_results = []
        # メモリ節約のため、解析時のみモデルをロードし、終わったらクリアする
        with st.spinner('解析中...（メモリ節約モード）'):
            # model_type='cyto2' の方が精度が高く、かつ安定する場合があります
            model = models.CellposeModel(gpu=False, model_type='cyto', device=torch.device('cpu'))
            
            for name, img in images_to_process:
                st.subheader(f"解析中: {name}")
                # チャンネル設定を工夫
                masks, _, _ = model.eval(img, diameter=target_diameter, channels=[0,0])
                
                props = measure.regionprops_table(masks, properties=['label', 'area', 'perimeter', 'equivalent_diameter'])
                df = pd.DataFrame(props)
                
                if not df.empty:
                    df['circularity'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2)
                    df['diameter_um'] = df['equivalent_diameter'] * um_per_pixel
                    df['filename'] = name
                    df_clean = df[df['circularity'] > circularity_threshold].copy()
                    all_results.append(df_clean)

                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.imshow(img)
                    ax.contour(masks > 0, colors='lime', linewidths=0.5)
                    st.pyplot(fig)
                    plt.close(fig) # メモリ解放
                    st.write(f"検出: {len(df_clean)}個 / 平均: {df_clean['diameter_um'].mean():.2f} μm")

        if all_results:
            final_df = pd.concat(all_results)
            output = python_io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                final_df.to_excel(writer, index=False)
            st.download_button("📊 Excel保存", output.getvalue(), "results.xlsx")
