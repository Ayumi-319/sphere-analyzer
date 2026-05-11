import streamlit as st
from cellpose import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure, segmentation
import io as python_io
from PIL import Image, ImageOps
import torch
import cv2
import gc

st.set_page_config(page_title="Sphere Analyzer v2.11", layout="wide")
st.title("🔴 スフェア自動計測ツール v2.11 (ぼかし機能追加版)")

@st.cache_resource
def get_model(m_type):
    return models.CellposeModel(gpu=False, model_type=m_type, device=torch.device('cpu'))

if 'masks_cache' not in st.session_state:
    st.session_state.masks_cache = {}

with st.sidebar:
    st.header("1. 基本設定")
    mag = st.radio("倍率:", ("4x", "10x", "カスタム"), index=0)
    um_per_pixel = 3.23 if mag == "4x" else 1.28 if mag == "10x" else st.number_input("μm/px", value=1.0)

    with st.form("ai_settings_form"):
        st.header("2. AIを助ける前処理")
        invert_image = st.checkbox("画像を白黒反転する", value=True)
        # 新機能：ぼかし（ガウスフィルター）
        blur_strength = st.slider("ぼかし (内部模様を消す)", 0, 10, 3, help="数値を上げるとスフェア内部がのっぺりし、縁だけが残ります")
        
        st.header("3. AI解析設定")
        model_choice = st.radio("AIモデル:", ("cyto2", "nuclei"), index=0)
        target_diameter = st.number_input("予想直径 (px)", value=100)
        flow_threshold = st.slider("切り離し強度", 0.0, 1.1, 0.9)
        cellprob_threshold = st.slider("検出感度", -6.0, 6.0, 0.0)
        
        submit_btn = st.form_submit_button("🚀 この設定でAI解析を実行")

    st.header("4. フィルタ設定 (即時反映)")
    exclude_border = st.checkbox("画像端を除外", value=True)
    circularity_threshold = st.slider("真円度しきい値", 0.0, 1.0, 0.7)

uploaded_files = st.file_uploader("ドロップ", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        img_raw = Image.open(f).convert('RGB')
        img_raw.thumbnail((800, 800), Image.Resampling.LANCZOS)
        
        img_edit = img_raw.copy()
        if invert_image:
            img_edit = ImageOps.invert(img_edit)
            
        img_np = np.array(img_edit)
        
        # ぼかし処理の適用
        if blur_strength > 0:
            k = blur_strength * 2 + 1  # カーネルサイズ（奇数にする）
            img_np = cv2.GaussianBlur(img_np, (k, k), 0)
        
        st.subheader(f"解析: {f.name}")
        col_pre, col_res = st.columns(2)
        
        with col_pre:
            # プレビューにもぼかし結果を表示
            st.image(img_np, caption="AIが実際に見ている画像（ぼかし適用後）", use_container_width=True, channels="RGB")

        cache_key = f"{f.name}_{invert_image}_{blur_strength}_{model_choice}_{target_diameter}_{flow_threshold}_{cellprob_threshold}"

        if submit_btn or cache_key in st.session_state.masks_cache:
            if cache_key not in st.session_state.masks_cache:
                with st.spinner('AIが計算中...'):
                    model_name = 'cyto2' if model_choice == "cyto2" else 'nuclei'
                    model = get_model(model_name)
                    
                    masks, _, _ = model.eval(img_np, 
                                             diameter=target_diameter, 
                                             flow_threshold=flow_threshold,
                                             cellprob_threshold=cellprob_threshold,
                                             channels=[0,0])
                    st.session_state.masks_cache[cache_key] = masks
                    gc.collect()

            masks = st.session_state.masks_cache[cache_key].copy()
            
            if exclude_border:
                masks = segmentation.clear_border(masks)
            
            props = measure.regionprops_table(masks, properties=['label', 'area', 'perimeter', 'equivalent_diameter'])
            df = pd.DataFrame(props)
            
            if not df.empty:
                df = df[df['perimeter'] > 0]
                df['circularity'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2)
                
                original_w, _ = Image.open(f).size
                current_w, _ = img_raw.size
                scale_factor = original_w / current_w
                df['diameter_um'] = df['equivalent_diameter'] * um_per_pixel * scale_factor
                
                df_clean = df[df['circularity'] > circularity_threshold].copy()
                
                with col_res:
                    fig, ax = plt.subplots()
                    ax.imshow(np.array(img_raw)) 
                    ax.contour(masks > 0, colors='lime', linewidths=0.5)
                    ax.axis('off')
                    st.pyplot(fig)
                    st.metric("検出数", f"{len(df_clean)} 個")
                    st.metric("平均直径", f"{df_clean['diameter_um'].mean():.1f} μm")
                    st.metric("平均真円度", f"{df_clean['circularity'].mean():.2f}")
            else:
                with col_res:
                    st.warning("スフェアが一つも検出されませんでした。")
