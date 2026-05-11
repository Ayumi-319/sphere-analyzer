import streamlit as st
from cellpose import models, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure
import io as python_io
from pptx import Presentation
from PIL import Image

st.set_page_config(page_title="Sphere Analyzer v2.0", layout="wide")

st.title("🔴 スフェア自動計測ツール v2.0")
st.write("画像ファイル（JPG/PNG）やパワポ（PPTX）をアップロードしてください。")

# サイドバー設定
with st.sidebar:
    st.header("1. 倍率設定")
    # 先週測定した値をセット
    mag = st.radio("顕微鏡の倍率を選択:", ("4x", "10x", "カスタム"))
    
    if mag == "4x":
        um_per_pixel = 3.23
    elif mag == "10x":
        um_per_pixel = 1.28
    else:
        um_per_pixel = st.number_input("μm/pixelを手入力", value=1.0)

    st.header("2. 解析パラメータ")
    target_diameter = st.slider("予想直径 (px)", 50, 300, 150)
    circularity_threshold = st.slider("真円度のしきい値", 0.5, 0.95, 0.8)

# ファイルアップロード（複数・パワポ対応）
uploaded_files = st.file_uploader("ファイルをドロップしてください（複数可）", type=['jpg', 'png', 'pptx'], accept_multiple_files=True)

all_results = []

if uploaded_files:
    # モデルの読み込み（初回のみ）
    with st.spinner('AIを準備中...'):
        model = models.CellposeModel(gpu=False, model_type='cyto')

    for uploaded_file in uploaded_files:
        images_to_process = []
        
        # パワポの場合
        if uploaded_file.name.endswith('.pptx'):
            prs = Presentation(uploaded_file)
            for i, slide in enumerate(prs.slides):
                for shape in slide.shapes:
                    if shape.shape_type == 13: # Picture
                        img_stream = python_io.BytesIO(shape.image.blob)
                        images_to_process.append((f"{uploaded_file.name}_slide{i+1}", io.imread(img_stream)))
        # 画像の場合
        else:
            images_to_process.append((uploaded_file.name, io.imread(uploaded_file)))

        # 解析実行
        for name, img in images_to_process:
            st.subheader(f"解析中: {name}")
            masks, _, _ = model.eval(img, diameter=target_diameter, channels=[0,0])
            
            props = measure.regionprops_table(masks, properties=['label', 'area', 'perimeter', 'equivalent_diameter'])
            df = pd.DataFrame(props)
            if not df.empty:
                df['circularity'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2)
                df['diameter_um'] = df['equivalent_diameter'] * um_per_pixel
                df['filename'] = name
                
                df_clean = df[df['circularity'] > circularity_threshold].copy()
                all_results.append(df_clean)

                # 表示
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.imshow(img)
                ax.contour(masks > 0, colors='lime', linewidths=0.5)
                st.pyplot(fig)
                st.write(f"検出数: {len(df_clean)} 個 / 平均直径: {df_clean['diameter_um'].mean():.2f} μm")

    # 全結果の統合とダウンロード
    if all_results:
        final_df = pd.concat(all_results)
        st.success("全ての解析が完了しました！")
        
        # Excelダウンロード
        output = python_io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            final_df.to_excel(writer, index=False, sheet_name='Summary')
        
        st.download_button(
            label="📊 解析結果をExcelでダウンロード",
            data=output.getvalue(),
            file_name="sphere_analysis_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
