import streamlit as st
from cellpose import models, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure
import io as python_io

st.set_page_config(page_title="iPS Sphere Analyzer", layout="wide")

st.title("🔴 iPSスフェア自動計測ツール")
st.write("画像をアップロードするだけで、AIがサイズを自動計測します。")

# サイドバーで設定
with st.sidebar:
    st.header("解析設定")
    target_diameter = st.slider("予想直径 (px)", 50, 300, 150)
    circularity_threshold = st.slider("真円度のしきい値", 0.5, 0.95, 0.8)
    um_per_pixel = st.number_input("μm/pixel", value=1.0)

uploaded_file = st.file_uploader("スフェアの画像を選択してください...", type=['jpg', 'png', 'tif'])

if uploaded_file is not None:
    # 画像の読み込み
    image = io.imread(uploaded_file)
    
    with st.spinner('AIが解析中...（初回は時間がかかります）'):
        # Cellposeの実行 (Webサーバー上ではCPUで動くことが多いです)
        model = models.CellposeModel(gpu=False, model_type='cyto')
        masks, flows, styles = model.eval(image, diameter=target_diameter, channels=[0,0])

        # 解析計算
        props = measure.regionprops_table(masks, properties=['label', 'area', 'perimeter', 'equivalent_diameter'])
        df = pd.DataFrame(props)
        df['circularity'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2)
        df['diameter_um'] = df['equivalent_diameter'] * um_per_pixel
        df['is_accepted'] = df['circularity'] > circularity_threshold
        df_clean = df[df['is_accepted']].copy()

    # 結果表示
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("解析結果画像")
        fig, ax = plt.subplots()
        ax.imshow(image)
        # 簡易的な縁取り表示
        ax.contour(masks > 0, colors='lime', linewidths=0.5)
        st.pyplot(fig)

    with col2:
        st.subheader("統計データ")
        st.metric("採用スフェア数", f"{len(df_clean)} 個")
        st.metric("平均直径", f"{df_clean['diameter_um'].mean():.2f} μm")
        st.dataframe(df_clean[['diameter_um', 'circularity']])

    # CSVダウンロード
    csv = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button("結果をCSVでダウンロード", csv, "result.csv", "text/csv")


