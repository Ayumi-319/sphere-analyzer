import streamlit as st
from cellpose import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure, segmentation
import io as python_io
from PIL import Image, ImageEnhance, ImageOps
import torch
import cv2
import gc
import traceback

st.set_page_config(page_title="Sphere Analyzer v2.9", layout="wide")
st.title("🔴 スフェア自動計測ツール v2.9 (白黒反転モード搭載)")

@st.cache_resource
def get_model(m_type):
    return models.CellposeModel(gpu=False, model_type=m_type, device=torch.device('cpu'))

with st.sidebar:
    st.header("1. 基本設定")
    mag = st.radio("倍率:", ("4x", "10x", "カスタム"), index=0)
    um_per_pixel = 3.23 if mag == "4x" else 1.28 if mag == "10x" else st.number_input("μm/px", value=1.0)

    st.header("2. AIを助ける前処理")
    # ここが今回の目玉です
    invert_image = st.checkbox("画像を白黒反転する (位相差に有効)", value=True)
    contrast = st.slider("コントラスト", 0.5, 3.0, 1.0) # 初期値を無加工に
    sharpness = st.slider("シャープネス", 0.0, 5.0, 1.0) # 初期値を無加工に
    
    st.header("3. AI解析設定")
    model_choice = st.radio("AIモデル選択:", ("cyto2", "nuclei"), index=0)
    target_diameter = st.number_input("予想直径 (px)", value=150)
    flow_threshold = st.slider("切り離し強度", 0.0, 1.1, 0.4)
    cellprob_threshold = st.slider("検出感度", -6.0, 6.0, 0.0)
    
    st.header("4. フィルタ設定 (即時)")
    exclude_border = st.checkbox("画像端を除外", value=True)
    circularity_threshold = st.slider("真円度しきい値", 0.0, 1.0, 0.7)

uploaded_files = st.file_uploader("ドロップ", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        try:
            # 画像の読み込みと軽量化
            img_raw = Image.open(f).convert('RGB')
            img_raw.thumbnail((800, 800), Image.Resampling.LANCZOS)
            
            # プレビュー用の画像処理
            img_edit = img_raw.copy()
            if invert_image:
                img_edit = ImageOps.invert(img_edit) # 白黒反転！
                
            enhancer_c = ImageEnhance.Contrast(img_edit)
            img_edit = enhancer_c.enhance(contrast)
            enhancer_s = ImageEnhance.Sharpness(img_edit)
            img_edit = enhancer_s.enhance(sharpness)
            
            img_np = np.array(img_edit)
            
            st.subheader(f"解析: {f.name}")
            col_pre, col_res = st.columns(2)
            with col_pre:
                st.image(img_edit, caption="AIに渡す画像（プレビュー）", use_container_width=True)
            
            if st.button(f"🚀 {f.name} のAI解析を開始", key=f"btn_{f.name}"):
                model_name = 'cyto2' if model_choice == "cyto2" else 'nuclei'
                model = get_model(model_name)
                
                with st.spinner('AIが計算中...'):
                    masks, _, _ = model.eval(img_np, 
                                             diameter=target_diameter, 
                                             flow_threshold=flow_threshold,
                                             cellprob_threshold=cellprob_threshold,
                                             channels=[0,0])
                    
                    if exclude_border:
                        masks = segmentation.clear_border(masks)
                    
                    props = measure.regionprops_table(masks, properties=['label', 'area', 'perimeter', 'equivalent_diameter'])
                    df = pd.DataFrame(props)
                    
                    if not df.empty:
                        df = df[df['perimeter'] > 0]
                        df['circularity'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2)
                        
                        # 縮小した分のスケールを補正して正しいμmを計算
                        original_w, _ = Image.open(f).size
                        current_w, _ = img_raw.size
                        scale_factor = original_w / current_w
                        df['diameter_um'] = df['equivalent_diameter'] * um_per_pixel * scale_factor
                        
                        df_clean = df[df['circularity'] > circularity_threshold].copy()
                        
                        with col_res:
                            fig, ax = plt.subplots()
                            # 結果の表示は元の色（反転していない状態）で表示
                            ax.imshow(np.array(img_raw)) 
                            ax.contour(masks > 0, colors='lime', linewidths=0.5)
                            ax.axis('off')
                            st.pyplot(fig)
                            st.metric("検出数", f"{len(df_clean)} 個")
                            st.metric("平均直径", f"{df_clean['diameter_um'].mean():.1f} μm")
                            st.metric("平均真円度", f"{df_clean['circularity'].mean():.2f}")
                    else:
                        st.warning("スフェアが一つも検出されませんでした。")
                
                del masks
                gc.collect()

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
            st.code(traceback.format_exc())
