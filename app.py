import streamlit as st
from cellpose import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure
import io as python_io
from pptx import Presentation
from PIL import Image
import torch
import cv2

st.set_page_config(page_title="Sphere Analyzer v2.3", layout="wide")
st.title("🔴 スフェア自動計測ツール v2.3 (高速版)")

# モデルのキャッシュ化（起動時1回のみ）
@st.cache_resource
def get_model():
    return models.CellposeModel(gpu=False, model_type='cyto', device=torch.device('cpu'))

with st.sidebar:
    st.header("1. 倍率設定")
    mag = st.radio("倍率:", ("4x", "10x", "カスタム"), index=0)
    um_per_pixel = 3.23 if mag == "4x" else 1.28 if mag == "10x" else st.number_input("μm/px", value=1.0)

    st.header("2. AI解析設定")
    target_diameter = st.number_input("予想直径 (px) ※変更すると再解析", value=150)
    
    st.header("3. フィルタ設定")
    circularity_threshold = st.slider("真円度しきい値 ※即時反映", 0.0, 1.0, 0.8)

uploaded_files = st.file_uploader("JPG/PNG/PPTXをドロップ", type=['jpg', 'png', 'pptx', 'jpeg'], accept_multiple_files=True)

# 解析結果を保存しておく箱
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}

if uploaded_files:
    model = get_model()
    all_final_results = []

    for f in uploaded_files:
        # 画像の読み込み処理
        if f.name.lower().endswith('.pptx'):
            prs = Presentation(f)
            images = [(f"{f.name}_S{i+1}", np.array(Image.open(python_io.BytesIO(s.image.blob)))) 
                      for i, sl in enumerate(prs.slides) for s in sl.shapes if s.shape_type == 13]
        else:
            images = [(f.name, np.array(Image.open(f)))]

        for name, img in images:
            st.subheader(f"解析: {name}")
            
            # AI解析（直径設定が変わった時だけ実行）
            cache_key = f"{name}_{target_diameter}"
            if cache_key not in st.session_state.analysis_cache:
                with st.spinner(f'{name} を計算中...'):
                    # 高速化のため画像をリサイズして解析（内部処理のみ）
                    h, w = img.shape[:2]
                    resized_img = cv2.resize(img, (w//2, h//2))
                    masks, _, _ = model.eval(resized_img, diameter=target_diameter//2, channels=[0,0])
                    # マスクを元のサイズに復元
                    masks = cv2.resize(masks.astype(np.uint16), (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    # 全データの計測
                    props = measure.regionprops_table(masks, properties=['label', 'area', 'perimeter', 'equivalent_diameter'])
                    st.session_state.analysis_cache[cache_key] = (pd.DataFrame(props), masks)

            full_df, masks = st.session_state.analysis_cache[cache_key]
            
            # フィルタリング（ここから下は一瞬で終わる）
            df = full_df.copy()
            df['circularity'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2)
            df['diameter_um'] = df['equivalent_diameter'] * um_per_pixel
            df['filename'] = name
            df_clean = df[df['circularity'] > circularity_threshold].copy()
            all_final_results.append(df_clean)

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
                st.metric("平均直径", f"{df_clean['diameter_um'].mean():.1f} μm")
                st.metric("平均真円度", f"{df_clean['circularity'].mean():.2f}")

    if all_final_results:
        final_df = pd.concat(all_final_results)
        output = python_io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            final_df.to_excel(writer, index=False)
        st.download_button("📊 結果をExcel保存", output.getvalue(), "results.xlsx")
