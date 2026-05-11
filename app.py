import streamlit as st
from cellpose import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure, segmentation, exposure
import io as python_io
from pptx import Presentation
from PIL import Image
import torch
import cv2

st.set_page_config(page_title="Sphere Analyzer v2.6", layout="wide")
st.title("🔴 スフェア自動計測ツール v2.6")

@st.cache_resource
def get_model():
    # cyto2 モデルに変更（より高精度な境界分離）
    return models.CellposeModel(gpu=False, model_type='cyto2', device=torch.device('cpu'))

with st.sidebar:
    st.header("1. 基本設定")
    mag = st.radio("倍率:", ("4x", "10x", "カスタム"), index=0)
    um_per_pixel = 3.23 if mag == "4x" else 1.28 if mag == "10x" else st.number_input("μm/px", value=1.0)

    st.header("2. 画像補正 (再計算)")
    contrast_factor = st.slider("コントラスト倍率", 0.0, 2.0, 1.0)
    
    st.header("3. AI解析設定 (再計算)")
    target_diameter = st.number_input("予想直径 (px)", value=150)
    flow_threshold = st.slider("切り離し強度 (Flow)", 0.0, 1.1, 0.4)
    # Cell probabilityの調整を追加：低いほどたくさん拾い、高いほど厳選する
    cellprob_threshold = st.slider("検出感度 (Cellprob)", -6.0, 6.0, 0.0, help="値を下げると、より多くのスフェアを拾おうとします")
    
    st.header("4. フィルタ設定 (即時反映)")
    exclude_border = st.checkbox("画像端の個体を除外", value=True)
    circularity_threshold = st.slider("真円度しきい値", 0.0, 1.0, 0.8)

uploaded_files = st.file_uploader("ドロップ", type=['jpg', 'png', 'pptx', 'jpeg'], accept_multiple_files=True)

if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}

if uploaded_files:
    model = get_model()
    all_final_results = []

    for f in uploaded_files:
        images = [(f.name, np.array(Image.open(f)))] # シンプルな画像読み込みのみに一旦集約

        for name, original_img in images:
            st.subheader(f"解析: {name}")
            
            cache_key = f"{name}_{contrast_factor}_{target_diameter}_{flow_threshold}_{cellprob_threshold}"
            if cache_key not in st.session_state.analysis_cache:
                with st.spinner('新モデルで解析中...'):
                    p2, p98 = np.percentile(original_img, (2 * contrast_factor, 98))
                    img_adj = exposure.rescale_intensity(original_img, in_range=(p2, p98))
                    
                    h, w = img_adj.shape[:2]
                    # 解析精度を上げるため、リサイズを緩やかに（少し大きく維持）
                    scale = 1000 / max(h, w) if max(h, w) > 1000 else 1.0
                    resized_img = cv2.resize(img_adj, (int(w*scale), int(h*scale)))
                    
                    masks, _, _ = model.eval(resized_img, 
                                             diameter=target_diameter*scale, 
                                             flow_threshold=flow_threshold,
                                             cellprob_threshold=cellprob_threshold,
                                             channels=[0,0])
                    
                    masks = cv2.resize(masks.astype(np.uint16), (w, h), interpolation=cv2.INTER_NEAREST)
                    if exclude_border:
                        masks = segmentation.clear_border(masks)
                        
                    props = measure.regionprops_table(masks, properties=['label', 'area', 'perimeter', 'equivalent_diameter'])
                    st.session_state.analysis_cache[cache_key] = (pd.DataFrame(props), masks)

            full_df, masks = st.session_state.analysis_cache[cache_key]
            df = full_df.copy()
            if not df.empty:
                # エラー回避用：perimeterが0のものを除外
                df = df[df['perimeter'] > 0]
                df['circularity'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2)
                df['diameter_um'] = df['equivalent_diameter'] * um_per_pixel
                df['filename'] = name
                df_clean = df[df['circularity'] > circularity_threshold].copy()
                all_final_results.append(df_clean)

                col1, col2 = st.columns([2, 1])
                with col1:
                    fig, ax = plt.subplots()
                    ax.imshow(original_img)
                    ax.contour(masks > 0, colors='lime', linewidths=0.5)
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)
                with col2:
                    st.metric("検出数", f"{len(df_clean)} 個")
                    st.metric("平均直径", f"{df_clean['diameter_um'].mean():.1f} μm")
                    st.metric("平均真円度", f"{df_clean['circularity'].mean():.2f}")
