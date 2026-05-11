import streamlit as st
from cellpose import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure, segmentation, exposure
import io as python_io
from PIL import Image, ImageEnhance
import torch
import cv2
import gc  # メモリ掃除用のお掃除ロボット
import traceback

st.set_page_config(page_title="Sphere Analyzer v2.8", layout="wide")
st.title("🔴 スフェア自動計測ツール v2.8 (軽量・安定版)")

@st.cache_resource
def get_model(m_type):
    return models.CellposeModel(gpu=False, model_type=m_type, device=torch.device('cpu'))

with st.sidebar:
    st.header("1. 画像補正 (高速プレビュー)")
    contrast = st.slider("コントラスト", 0.5, 3.0, 1.5)
    sharpness = st.slider("シャープネス", 0.0, 5.0, 2.0)
    
    st.header("2. AI解析設定 (要・再計算)")
    model_choice = st.radio("AIモデル選択:", ("cyto2", "nuclei"), index=1)
    target_diameter = st.number_input("予想直径 (px)", value=80)
    flow_threshold = st.slider("切り離し強度", 0.0, 1.1, 0.9)
    cellprob_threshold = st.slider("検出感度", -6.0, 6.0, 0.0)
    
    st.header("3. フィルタ設定 (即時)")
    exclude_border = st.checkbox("画像端を除外", value=True)
    circularity_threshold = st.slider("真円度しきい値", 0.0, 1.0, 0.7)

uploaded_files = st.file_uploader("ドロップ", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        try:
            # 1. 画像を開いた瞬間に強制的にリサイズしてメモリを節約！ (長辺最大800px)
            img_raw = Image.open(f).convert('RGB')
            img_raw.thumbnail((800, 800), Image.Resampling.LANCZOS)
            
            # 2. 補正処理
            enhancer_c = ImageEnhance.Contrast(img_raw)
            img_edit = enhancer_c.enhance(contrast)
            enhancer_s = ImageEnhance.Sharpness(img_edit)
            img_edit = enhancer_s.enhance(sharpness)
            
            img_np = np.array(img_edit)
            
            st.subheader(f"画像プレビュー: {f.name}")
            col_pre, col_res = st.columns(2)
            with col_pre:
                st.image(img_edit, caption="AIに渡す画像", use_container_width=True)
            
            if st.button(f"🚀 {f.name} のAI解析を開始", key=f"btn_{f.name}"):
                model_name = 'cyto2' if model_choice == "cyto2" else 'nuclei'
                model = get_model(model_name)
                
                with st.spinner('AIが計算中...（数分かかる場合があります）'):
                    # AI解析実行
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
                        df_clean = df[df['circularity'] > circularity_threshold].copy()
                        
                        with col_res:
                            fig, ax = plt.subplots()
                            ax.imshow(np.array(img_raw))
                            ax.contour(masks > 0, colors='lime', linewidths=0.5)
                            ax.axis('off')
                            st.pyplot(fig)
                            st.metric("検出数", f"{len(df_clean)} 個")
                    else:
                        st.warning("スフェアが一つも検出されませんでした。")
                
                # 使い終わったメモリを強制的にお掃除
                del masks
                gc.collect()

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
            st.code(traceback.format_exc())
