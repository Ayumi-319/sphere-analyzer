import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
from cellpose import models
from skimage import measure, segmentation

st.set_page_config(page_title="Sphere Analyzer AI v6.2", layout="wide")
st.title("🤖 スフェア自動計測ツール v6.2 (AI修正版)")

@st.cache_resource
def load_ai_model():
    import torch
    device = torch.device('cpu')
    # ここを修正：最新のCellposeは models.CellposeModel または models.Cellpose を使用します
    return models.Cellpose(model_type='cyto', gpu=False, device=device)

with st.sidebar:
    st.header("1. スケール設定")
    mag = st.radio("倍率:", ("4x", "10x", "20x"), index=1)
    um_per_pixel = {"4x": 1.9109, "10x": 0.7643, "20x": 0.3817}.get(mag, 1.0)

    st.header("2. AI解析設定")
    target_diam = st.number_input("予想直径 (px) ※0で自動", value=0)
    chan_threshold = st.slider("検出感度", -6.0, 6.0, 0.0)
    flow_threshold = st.slider("切り離し強度", 0.0, 1.0, 0.4)

    st.header("3. フィルタ ＆ 表示")
    exclude_border = st.checkbox("画像端を除外", value=True)
    min_area = st.number_input("最小面積 (px)", value=500)
    show_numbers = st.checkbox("番号を表示", value=True)

uploaded_file = st.file_uploader("画像をアップロード", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img_raw = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(img_raw)
    
    if st.button("🚀 AI解析を開始"):
        with st.spinner("AIが計算中..."):
            try:
                model = load_ai_model()
                # channels=[0,0] はグレースケール用
                masks, flows, styles, diams = model.eval(
                    img_np, 
                    diameter=None if target_diam == 0 else target_diam,
                    channels=[0, 0],
                    flow_threshold=flow_threshold,
                    cellprob_threshold=chan_threshold,
                    resample=False
                )
                
                if exclude_border:
                    masks = segmentation.clear_border(masks)
                
                st.session_state.ai_masks = masks
                st.success("解析完了！")
            except Exception as e:
                st.error(f"解析中にエラーが発生しました: {e}")

    if 'ai_masks' in st.session_state:
        masks = st.session_state.ai_masks
        props = measure.regionprops(masks)
        col_img, col_res = st.columns([1.5, 1])
        
        final_list = []
        fig, ax = plt.subplots()
        ax.imshow(img_np)
        
        idx = 1
        for p in props:
            if p.area >= min_area:
                if show_numbers:
                    ax.text(p.centroid[1], p.centroid[0], str(idx), color='red', fontsize=7, fontweight='bold', ha='center')
                final_list.append({
                    'No': idx,
                    '直径(μm)': p.equivalent_diameter * um_per_pixel,
                    '面積(px)': p.area
                })
                idx += 1
        
        ax.contour(masks > 0, colors='lime', linewidths=0.5)
        ax.axis('off')
        
        with col_img:
            st.pyplot(fig)
        with col_res:
            df = pd.DataFrame(final_list)
            st.metric("検出数", f"{len(df)} 個")
            if not df.empty:
                st.dataframe(df, height=400)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("結果をCSV保存", csv, "ai_result.csv")
