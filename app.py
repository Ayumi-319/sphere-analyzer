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

# ページ設定
st.set_page_config(page_title="Sphere Analyzer v2.2", layout="wide")

st.title("🔴 スフェア自動計測ツール v2.2")

# サイドバー設定
with st.sidebar:
    st.header("1. 倍率設定")
    mag = st.radio("顕微鏡の倍率を選択:", ("4x", "10x", "カスタム"), index=0)
    if mag == "4x": um_per_pixel = 3.23
    elif mag == "10x": um_per_pixel = 1.28
    else: um_per_pixel = st.number_input("μm/pixelを手入力", value=1.0)

    st.header("2. 解析パラメータ")
    target_diameter = st.slider("予想直径 (px)", 50, 300, 150)
    circularity_threshold = st.slider("真円度のしきい値", 0.5, 0.95, 0.8)

uploaded_files = st.file_uploader("JPG/PNG/PPTXをドロップ", type=['jpg', 'png', 'pptx', 'jpeg'], accept_multiple_files=True)

# セッション状態（キャッシュ）の初期化：倍率変更時にAIを再走させないため
if 'masks_cache' not in st.session_state:
    st.session_state.masks_cache = {}

if uploaded_files:
    images_to_process = []
    for f in uploaded_files:
        if f.name.lower().endswith('.pptx'):
            prs = Presentation(f)
            for i, slide in enumerate(prs.slides):
                for shape in slide.shapes:
                    if shape.shape_type == 13:
                        img_stream = python_io.BytesIO(shape.image.blob)
                        images_to_process.append((f"{f.name}_S{i+1}", np.array(Image.open(img_stream))))
        else:
            images_to_process.append((f.name, np.array(Image.open(f))))

    if images_to_process:
        all_results = []
        # AIモデルのロード（キャッシュを利用して高速化）
        @st.cache_resource
        def load_model():
            return models.CellposeModel(gpu=False, model_type='cyto', device=torch.device('cpu'))

        model = load_model()

        for name, img in images_to_process:
            st.subheader(f"解析: {name}")
            
            # パラメータが変わった時だけAI解析を実行
            cache_key = f"{name}_{target_diameter}"
            if cache_key not in st.session_state.masks_cache:
                with st.spinner(f'{name} をAI解析中...'):
                    masks, _, _ = model.eval(img, diameter=target_diameter, channels=[0,0])
                    st.session_state.masks_cache[cache_key] = masks
            
            masks = st.session_state.masks_cache[cache_key]
            
            # 計測処理
            props = measure.regionprops_table(masks, properties=['label', 'area', 'perimeter', 'equivalent_diameter'])
            df = pd.DataFrame(props)
            
            if not df.empty:
                df['circularity'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2)
                df['diameter_um'] = df['equivalent_diameter'] * um_per_pixel
                df['filename'] = name
                df_clean = df[df['circularity'] > circularity_threshold].copy()
                all_results.append(df_clean)

                # 表示用
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig, ax = plt.subplots()
                    ax.imshow(img)
                    ax.contour(masks > 0, colors='lime', linewidths=0.5)
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)
                
                with col2:
                    st.metric("検出数", f"{len(df_clean)} 個")
                    st.metric("平均直径", f"{df_clean['diameter_um'].mean():.2f} μm")
                    st.metric("平均真円度", f"{df_clean['circularity'].mean():.3f}")
            else:
                st.warning(f"{name}: 検出なし")

        if all_results:
            final_df = pd.concat(all_results)
            output = python_io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                final_df.to_excel(writer, index=False)
            st.download_button("📊 全結果をExcelで保存", output.getvalue(), "sphere_results.xlsx")
