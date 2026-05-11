import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import color, filters, morphology, feature, segmentation, measure
from scipy import ndimage as ndi
from PIL import Image, ImageEnhance, ImageOps
import cv2
from cellpose import models
import torch
import gc

st.set_page_config(page_title="Sphere Analyzer v4.0", layout="wide")
st.title("🔴 スフェア自動計測ツール v4.0 (ImageJ＋AI ハイブリッド版)")

@st.cache_resource
def get_model(m_type):
    return models.CellposeModel(gpu=False, model_type=m_type, device=torch.device('cpu'))

if 'ai_masks_cache' not in st.session_state:
    st.session_state.ai_masks_cache = {}

with st.sidebar:
    st.header("1. 基本設定")
    mag = st.radio("倍率:", ("4x", "10x", "カスタム"), index=0)
    um_per_pixel = 3.23 if mag == "4x" else 1.28 if mag == "10x" else st.number_input("μm/px", value=1.0)

    st.header("2. シルエットの作成 (即時)")
    invert_image = st.checkbox("画像を白黒反転する", value=False)
    contrast = st.slider("コントラスト", 0.5, 3.0, 1.0)
    sharpness = st.slider("シャープネス", 0.0, 5.0, 1.0)

    extraction_mode = st.radio("抽出方法:", ("背景を一気にくり抜く (おすすめ)", "細かい影を拾う (従来)"), index=0)
    blur_sigma = st.slider("ぼかし強さ", 0.0, 10.0, 0.0)
    
    if extraction_mode == "背景を一気にくり抜く (おすすめ)":
        global_offset = st.slider("背景の判定ライン (閾値微調整)", -0.30, 0.30, -0.13, step=0.01)
    else:
        block_size_slider = st.slider("二値化の細かさ (Block Size)", 11, 201, 51, step=10)
        local_offset = st.slider("縁の拾いやすさ (Offset)", -0.10, 0.10, 0.00, step=0.01)

    st.write("💡 スライダー右端の「数字」をクリックすると直接手入力できます")
    remove_white = st.slider("⚪ 白いゴミ取り (背景ノイズ)", 0, 1000, 0, step=10)
    remove_black = st.slider("⚫ 黒いゴミ取り (内部の穴埋め)", 0, 200, 20, step=1)

    st.header("3. 解析エンジンの選択")
    engine = st.radio("エンジン:", ("ImageJ (爆速・調整用)", "AI Cellpose (低速・丸く仕上げる)"), index=0)
    
    if engine == "ImageJ (爆速・調整用)":
        min_dist = st.number_input("中心間の最小距離 (px)", value=30)
    else:
        with st.form("ai_form"):
            st.info("💡 上で作った「白黒のシルエット」をAIに渡し、綺麗な丸に整えさせます。")
            ai_model_choice = st.radio("AIモデル:", ("cyto2", "nuclei"), index=0)
            ai_diameter = st.number_input("予想直径 (px) ※0で自動", value=0)
            ai_flow = st.slider("切り離し強度", 0.0, 1.1, 0.9)
            ai_prob = st.slider("検出感度", -6.0, 6.0, 0.0)
            run_ai_btn = st.form_submit_button("🚀 AIで仕上げる")

    st.header("4. 最終フィルタ設定")
    exclude_border = st.checkbox("画像端を除外", value=True)
    min_area = st.number_input("最小面積 (px)", value=100)
    circularity_threshold = st.slider("真円度しきい値", 0.0, 1.0, 0.20)

uploaded_files = st.file_uploader("ドロップ", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        img_raw = Image.open(f).convert('RGB')
        img_raw.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        
        img_edit = img_raw.copy()
        if invert_image:
            img_edit = ImageOps.invert(img_edit)
            
        enhancer_c = ImageEnhance.Contrast(img_edit)
        img_edit = enhancer_c.enhance(contrast)
        enhancer_s = ImageEnhance.Sharpness(img_edit)
        img_edit = enhancer_s.enhance(sharpness)
        
        img_np = np.array(img_edit)
        gray = color.rgb2gray(img_np)
        
        if blur_sigma > 0:
            blurred = filters.gaussian(gray, sigma=blur_sigma)
        else:
            blurred = gray

        if extraction_mode == "背景を一気にくり抜く (おすすめ)":
            base_thresh = filters.threshold_otsu(blurred)
            adj_thresh = base_thresh + global_offset
            if invert_image:
                binary = blurred > adj_thresh
            else:
                binary = blurred < adj_thresh
        else:
            block_size = int(block_size_slider)
            if block_size % 2 == 0:
                block_size += 1
            local_thresh = filters.threshold_local(blurred, block_size, offset=local_offset)
            if invert_image:
                binary = blurred > local_thresh
            else:
                binary = blurred < local_thresh 
            
        if remove_white > 0:
            cleaned = morphology.remove_small_objects(binary, min_size=remove_white)
        else:
            cleaned = binary

        if remove_black > 0:
            filled = morphology.remove_small_holes(cleaned, area_threshold=remove_black)
        else:
            filled = cleaned

        labels = None
        
        # --- エンジンによる処理の分岐 ---
        if engine == "ImageJ (爆速・調整用)":
            distance = ndi.distance_transform_edt(filled)
            coords = feature.peak_local_max(distance, min_distance=min_dist, labels=filled)
            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(coords.T)] = True
            markers, _ = ndi.label(mask)
            labels = segmentation.watershed(-distance, markers, mask=filled)
            
        else:
            cache_key = f"{f.name}_{invert_image}_{contrast}_{sharpness}_{extraction_mode}_{blur_sigma}_{global_offset if extraction_mode == '背景を一気にくり抜く (おすすめ)' else f'{block_size_slider}_{local_offset}'}_{remove_white}_{remove_black}_{ai_model_choice}_{ai_diameter}_{ai_flow}_{ai_prob}"
            
            # AIに渡す画像を「純粋な白黒シルエット」に差し替える
            ai_input_img = (filled.astype(np.uint8) * 255)
            
            if run_ai_btn or cache_key in st.session_state.ai_masks_cache:
                if cache_key not in st.session_state.ai_masks_cache:
                    with st.spinner('AIがシルエットを丸く整えながら計算中...'):
                        model_name = 'cyto2' if ai_model_choice == "cyto2" else 'nuclei'
                        model = get_model(model_name)
                        diam = None if ai_diameter == 0 else ai_diameter
                        
                        masks, _, _ = model.eval(ai_input_img, 
                                                 diameter=diam, 
                                                 flow_threshold=ai_flow,
                                                 cellprob_threshold=ai_prob,
                                                 channels=[0,0])
                        st.session_state.ai_masks_cache[cache_key] = masks
                        gc.collect()
                labels = st.session_state.ai_masks_cache[cache_key].copy()

        st.subheader(f"解析: {f.name}")
        
        with st.expander("🔍 AIに渡すシルエット（白黒）を見る"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(img_edit, caption="元の画像", use_container_width=True)
            with col_b:
                st.image(filled.astype(float), caption="AIの脳内に見える画像 (この白塊をAIが丸く整えます)", use_container_width=True)

        if labels is not None:
            if exclude_border:
                labels = segmentation.clear_border(labels)

            props = measure.regionprops_table(labels, properties=['label', 'area', 'perimeter', 'equivalent_diameter'])
            df = pd.DataFrame(props)
            
            col_res1, col_res2 = st.columns([2, 1])
            
            if not df.empty:
                df = df[df['area'] >= min_area]
                df = df[df['perimeter'] > 0]
                df['circularity'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2)
                
                original_w, _ = Image.open(f).size
                current_w, _ = img_raw.size
                scale_factor = original_w / current_w
                df['diameter_um'] = df['equivalent_diameter'] * um_per_pixel * scale_factor
                
                df_clean = df[df['circularity'] > circularity_threshold].copy()
                
                with col_res2:
                    show_outline = st.checkbox("🟢 緑の線を表示する", value=True)
                    st.metric("検出数", f"{len(df_clean)} 個")
                    if len(df_clean) > 0:
                        st.metric("平均直径", f"{df_clean['diameter_um'].mean():.1f} μm")
                        st.metric("平均真円度", f"{df_clean['circularity'].mean():.2f}")
                        
                with col_res1:
                    fig, ax = plt.subplots()
                    ax.imshow(img_np) 
                    
                    if show_outline:
                        ax.contour(labels > 0, colors='lime', linewidths=0.5)
                    
                    ax.axis('off')
                    st.pyplot(fig)
                    
            else:
                with col_res2:
                    st.warning("検出されませんでした。")
        else:
             st.info("👆 左のメニューから「🚀 AIで仕上げる」ボタンを押してください。")
