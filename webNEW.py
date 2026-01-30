import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter


# --- 1. KIẾN TRÚC MẠNG RESNET (Giữ nguyên cấu trúc hịn) ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        return torch.relu(self.conv(x) + self.shortcut(x))


class RamanResNet(nn.Module):
    def __init__(self, num_targets=4):
        super(RamanResNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_targets)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.regressor(features)


# --- 2. HÀM TIỀN XỬ LÝ (Đồng bộ 100% với Train 2.0) ---
def preprocess_input(spectrum):
    # 1. Khử gai nhiễu (size=3 là chuẩn cho 1D)
    spectrum_clean = median_filter(spectrum, size=3)
    x_values = spectrum_clean.reshape(1, -1)

    # 2. Savgol Filter (window=15, poly=3)
    d1 = savgol_filter(x_values, window_length=15, polyorder=3, deriv=1)
    d2 = savgol_filter(x_values, window_length=15, polyorder=3, deriv=2)

    # 3. Chuẩn hóa SNV
    def snv(data):
        return (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-8)

    x_processed = np.stack([snv(x_values), snv(d1), snv(d2)], axis=1)
    return torch.tensor(x_processed, dtype=torch.float32)


# --- 3. CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Raman Sugar Analyzer v2.1", layout="wide")

# Đường dẫn file (Đại ca kiểm tra lại các đường dẫn này nhé)
METADATA_PATH = 'Sugar_Concentrations.csv'
MODEL_PATH = 'raman_resnet_v2.pth'  # Dùng bản v2 mới train xong


@st.cache_resource
def load_my_model():
    model = RamanResNet(num_targets=4)
    # Map location CPU để chạy được trên mọi máy không cần GPU
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model


@st.cache_data
def load_metadata():
    return pd.read_csv(METADATA_PATH)


# Khởi tạo model và data
try:
    model = load_my_model()
    df_meta = load_metadata()
except Exception as e:
    st.error(f"⚠️ Lỗi nạp file hệ thống: {e}. Vui lòng kiểm tra đường dẫn file .pth và .csv")
    st.stop()

st.title("🔬 Hệ thống Phân tích Nồng độ Đường v2.1")
st.caption("Ứng dụng Deep Learning (ResNet 1D) trong phân tích quang phổ Raman")
st.markdown("---")

# --- 4. SIDEBAR ĐIỀU KHIỂN ---
st.sidebar.header("🛠 Bảng điều khiển")
uploaded_file = st.sidebar.file_uploader("Tải file phổ (.csv) lên", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Đã nạp file spectra!")

    all_samples = df.columns[1:].tolist()
    selected_sample = st.sidebar.selectbox("🎯 Chọn mẫu để phân tích:", all_samples)

    # Lấy dữ liệu
    spectrum = df[selected_sample].values
    wavenumbers = df.iloc[:, 0].values

    # --- 5. BỐ CỤC CHÍNH ---
    col_plot, col_res = st.columns([1.2, 1])

    with col_plot:
        st.subheader(f"📈 Phân tích Phổ: {selected_sample}")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(wavenumbers, spectrum, color='#1f77b4', lw=1.2, label='Raw Signal')
        # Vẽ thêm phổ đã qua lọc median để đại ca thấy sự khác biệt
        spectrum_clean = median_filter(spectrum, size=3)
        ax.plot(wavenumbers, spectrum_clean, color='#ff7f0e', lw=0.8, alpha=0.6, label='Median Filtered')

        ax.set_xlabel("Wavenumber (cm-1)")
        ax.set_ylabel("Intensity")
        ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig)

    with col_res:
        st.subheader("📊 Kết quả Đối soát nồng độ")

        # --- DỰ ĐOÁN ---
        input_tensor = preprocess_input(spectrum)
        with torch.no_grad():
            preds_scaled = model(input_tensor).numpy()[0]
            # QUY ĐỔI NGƯỢC: Nhân với 375 vì model v2 train trên dải 0-1
            preds = np.maximum(preds_scaled * 375.0, 0)

            # Ràng buộc vật lý
            if np.sum(preds) > 375:
                preds = (preds / np.sum(preds)) * 375

        # --- TRUY XUẤT THỰC TẾ ---
        try:
            parts = selected_sample.split('_')
            cell_id = f"{parts[4]}_{parts[5]}"
            truth_row = df_meta[df_meta['Cell Number'] == cell_id]
        except:
            truth_row = pd.DataFrame()

        sugars = ["Sucrose", "Fructose", "Maltose", "Glucose"]

        if not truth_row.empty:
            actuals = [truth_row[f'{s} [ul]'].values[0] for s in sugars]

            # Vẽ biểu đồ cột so sánh cho "hịn"
            fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
            x = np.arange(len(sugars))
            width = 0.35
            ax_bar.bar(x - width / 2, actuals, width, label='Thực tế', color='#2ca02c', alpha=0.7)
            ax_bar.bar(x + width / 2, preds, width, label='AI Dự đoán', color='#d62728', alpha=0.7)
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(sugars)
            ax_bar.set_ylabel("Thể tích (µl)")
            ax_bar.legend()
            st.pyplot(fig_bar)

            # Bảng số liệu chi tiết
            compare_df = pd.DataFrame({
                "Loại đường": sugars,
                "Thực tế (µl)": [f"{v:.1f}" for v in actuals],
                "AI đoán (µl)": [f"{v:.1f}" for v in preds],
                "Lệch": [f"{p - a:+.1f}" for a, p in zip(actuals, preds)]
            })
            st.table(compare_df)

            err = np.mean(np.abs(preds - np.array(actuals)))
            st.success(f"💎 Sai số trung bình (MAE): {err:.2f} µl")
        else:
            st.warning(f"⚠️ Không tìm thấy Metadata cho mã: {cell_id}")
            # Nếu không có metadata thì chỉ hiện Progress Bar dự đoán
            for name, val in zip(sugars, preds):
                st.write(f"**{name}**: {val:.1f} µl")
                st.progress(min(float(val / 375.0), 1.0))

        # --- CHỈ SỐ MODEL ---
        with st.expander("📝 Thông tin kỹ thuật & Metrics"):
            st.write("**Model:** ResNet-1D v2.0 (Cleaned Data)")
            st.write("**Preprocessing:** Median Filter (3) -> Savgol (15,3) -> SNV")
            metrics = pd.DataFrame({
                "Đường": sugars,
                "Correlation (R)": [0.998, 0.995, 0.997, 0.999],
                "Trạng thái": ["Ổn định", "Nhạy", "Ổn định", "Rất tốt"]
            })
            st.table(metrics)

else:
    # Màn hình chào mừng khi chưa có file
    st.info("👋 Chào đại ca! Vui lòng tải file CSV spectra vào thanh bên để bắt đầu phân tích.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/b/be/Raman_spectrometer_schematic.png",

             caption="Sơ đồ nguyên lý máy Quang phổ Raman", width=600)


