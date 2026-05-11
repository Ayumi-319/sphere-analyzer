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

st.set_page_config(page_title="Sphere Analyzer v2.10", layout="wide")
st.title("🔴 スフェア自動計測ツール v2.10 (パラメータ調整特化版)")

@st.cache_resource
def get_model(m_type):
    return models.CellposeModel(gpu=False, model_type=m_type, device=torch.device('cpu'))

if 'masks_cache' not in st.session_state:
    st.session_state.masks_cache = {}

with st.sidebar:
    st.header("1. 基本設定")
    mag = st.radio("倍率:", ("4x", "10x", "カスタム"), index=0)
    um_per_pixel = 3.23 if mag == "4x" else 1.28 if mag == "10x" else st.number_input("μm/px", value=1.0)

    # AIの設定を「フォーム」で囲み、ボタンを押すまで再計算が走らないようにする
    with st.form("ai_settings_form"):
        st.header("2. AI解析設定 (要・再計算)")
        invert_image = st.checkbox("画像を白黒反転する (必須)", value=True)
        model_choice = st.radio("AIモデル:", ("cyto2", "nuclei"), index=0)
        target_diameter = st.number_input("予想直径 (px)", value=100, help="実際の見た目に近いサイズにすると精度が上がります")
        
        # 初期値を高めに設定しました
        flow_threshold = st.slider("切り離し強度", 0.0, 1.1, 0.9, help="合体してしまう場合は0.9〜1.0に上げてください")
        cellprob_threshold = st.slider("検出感度", -6.0, 6.0, 0.0)
        
        submit_btn = st.form_submit_button("🚀 この設定でAI解析を実行")

    st.header("3. フィルタ設定 (即時反映)")
    exclude_border = st.checkbox("画像端を除外", value=True)
    circularity_threshold = st.slider("真円度しきい値", 0.0, 1.0, 0.7)

uploaded_files = st.file_uploader("ドロップ", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        # 画像の読み込みと軽量化
        img_raw = Image.open(f).convert('RGB')
        img_raw.thumbnail((800, 800), Image.Resampling.LANCZOS)
        
        img_edit = img_raw.copy()
        if invert_image:
            img_edit = ImageOps.invert(img_edit)
        img_np = np.array(img_edit)
        
        st.subheader(f"解析: {f.name}")
        col_pre, col_res = st.columns(2)
        
        with col_pre:
            st.image(img_edit, caption="AIが実際に見ている画像", use_container_width=True)

        # キャッシュキーを作成（設定が変わったら別のキーになる）
        cache_key = f"{f.name}_{invert_image}_{model_choice}_{target_diameter}_{flow_threshold}_{cellprob_threshold}"

        # 「解析ボタンが押された時」または「すでに今の設定で解析済みの時」に結果を表示
        if submit_btn or cache_key in st.session_state.masks_cache:
            
            # まだ解析していない設定ならAIを走らせる
            if cache_key not in st.session_state.masks_cache:
                with st.spinner('AIが計算中...'):
                    model_name = 'cyto2' if model_choice == "cyto2" else 'nuclei'
                    model = get_model(model_name)
                    
                    masks, _, _ = model.eval(img_np, 
                                             diameter=target_diameter, 
                                             flow_threshold=flow_threshold,
                                             cellprob_threshold=cellprob_threshold,
                                             channels=[0,0])
                    # 計算結果を保存
                    st.session_state.masks_cache[cache_key] = masks
                    gc.collect()

            # 保存してある結果（マスク）を呼び出す
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
