import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import io
from cellpose import models
import torch

st.set_page_config(page_title="Sphere Analyzer AI v6.0", layout="wide")
st.title("🤖 スフェア自動計測ツール v6.0 (AI搭載版)")

# --- AIモデルのロード (キャッシュして高速化) ---
@st.cache_resource
def load_ai_model():
    # 'cyto' モデルを使用（丸い物体の認識に強い）
    return models.Cellpose(gpu=False, model_type='cyto')

with st.sidebar:
    st.header("1. スケール設定")
    mag = st.radio("倍率:", ("4x", "10x", "20x"), index=1)
    um_per_pixel = {"4x": 1.9109, "10x": 0.7643, "20x": 0.3817}.get(mag, 1.0)

    st.header("2. AI解析設定")
    st.info("💡 古典的な補正は不要。AIが直接形を見抜きます。")
    # AIが探すべき物体の大きさ（px）
    target_diam = st.number_input("予想直径 (px) ※0で自動計算", value=0)
    # 検出の感度（高いほどたくさん拾う）
    chan_threshold = st.slider("検出感度 (Cellprob Threshold)", -6.0, 6.0, 0.0)
    # 切り離しの強さ（高いほど細かく分ける）
    flow_threshold = st.slider("切り離し強度 (Flow Threshold)", 0.0, 1.0, 0.4)

    st.header("3. フィルタ ＆ 表示")
    exclude_border = st.checkbox("画像端を除外", value=True)
    min_area = st.number_input("最小面積 (px)", value=500)
    show_numbers = st.checkbox("番号を表示", value=True)

uploaded_file = st.file_uploader("画像をアップロード", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img_raw = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(img_raw)
    
    # 解析実行ボタン
    if st.button("🚀 AI解析を開始（少し時間がかかります）"):
        with st.spinner("AIがスフェアの形を「思考」しています..."):
            model = load_ai_model()
            
            # AI解析実行
            # channels=[0,0] はグレースケールとして処理することを意味します
            masks, flows, styles, diams = model.eval(
                img_np, 
                diameter=None if target_diam == 0 else target_diam,
                channels=[0, 0],
                flow_threshold=flow_threshold,
                cellprob_threshold=chan_threshold,
                resample=True
            )
            
            # 画像端の除去
            if exclude_border:
                from skimage.segmentation import clear_border
                masks = clear_border(masks)
            
            st.session_state.ai_masks = masks
            st.success("解析完了！")

    # --- 結果表示 ---
    if 'ai_masks' in st.session_state:
        masks = st.session_state.ai_masks
        props = measure.regionprops(masks) if 'measure' in locals() else []
        # regionpropsのために追加インポートが必要な場合
        from skimage import measure
        props = measure.regionprops(masks)
        
        col_img, col_res = st.columns([1.5, 1])
        
        final_list = []
        fig, ax = plt.subplots()
        ax.imshow(img_np)
        
        idx = 1
        for p in props:
            if p.area >= min_area:
                # 番号表示
                if show_numbers:
                    ax.text(p.centroid[1], p.centroid[0], str(idx), color='red', fontsize=8, fontweight='bold', ha='center')
                
                final_list.append({
                    'No': idx,
                    '直径(μm)': p.equivalent_diameter * um_per_pixel,
                    '真円度': (4 * np.pi * p.area) / (p.perimeter ** 2) if p.perimeter > 0 else 0
                })
                idx += 1
        
        ax.contour(masks > 0, colors='lime', linewidths=0.8)
        ax.axis('off')
        
        with col_img:
            st.pyplot(fig)
        
        with col_res:
            df = pd.DataFrame(final_list)
            st.metric("AI検出数", f"{len(df)} 個")
            if not df.empty:
                st.metric("平均直径", f"{df['直径(μm)'].mean():.2f} μm")
                st.dataframe(df)
                st.download_button("CSV保存", df.to_csv(index=False).encode('utf-8'), "ai_result.csv")
