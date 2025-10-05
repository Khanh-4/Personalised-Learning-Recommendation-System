import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

# ===============================
# 🎯 Load mô hình & dữ liệu
# ===============================

RESULTS_DIR = "../results"  # hoặc "./results" tùy cấu trúc thư mục của bạn
DATA_PATH = os.path.join(RESULTS_DIR, "summary.csv")

@st.cache_data
def load_summary():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        return df
    else:
        st.warning("⚠️ Không tìm thấy summary.csv, hãy chạy main.py trước!")
        return pd.DataFrame()

summary_df = load_summary()

# ===============================
# 🌟 Header
# ===============================
st.set_page_config(page_title="Personalized Learning Recommendation Demo", layout="wide")
st.title("🎓 Personalized Learning Recommendation System")
st.markdown("### 💡 Step 9: Demo Interface (Streamlit)")

# ===============================
# 📊 Hiển thị tổng quan kết quả
# ===============================
st.subheader("📈 Evaluation Summary (from Step 14)")
if not summary_df.empty:
    st.dataframe(summary_df.style.highlight_max(axis=0))
else:
    st.info("Chưa có dữ liệu đánh giá, vui lòng chạy pipeline trước.")

# ===============================
# 🧍 Chọn user để gợi ý
# ===============================
st.subheader("🧩 Recommend for a User")
user_id = st.number_input("Nhập ID người dùng:", min_value=0, step=1, value=1)
top_k = st.slider("Số lượng gợi ý (Top K):", 1, 20, 5)

# ===============================
# 🔍 Hàm gợi ý giả lập (demo)
# ===============================
def get_recommendations(user_id, top_k=5):
    # ✅ Giả lập random, sau có thể thay bằng model thực (MF, CB, Meta, …)
    fake_items = [f"Course_{i}" for i in np.random.choice(range(100), top_k, replace=False)]
    fake_scores = np.random.rand(top_k)
    df = pd.DataFrame({
        "Recommended Item": fake_items,
        "Predicted Score": fake_scores
    }).sort_values(by="Predicted Score", ascending=False)
    return df

# ===============================
# 🪄 Hiển thị gợi ý
# ===============================
if st.button("🚀 Generate Recommendations"):
    recs = get_recommendations(user_id, top_k)
    st.success(f"✅ Top {top_k} recommendations for user {user_id}")
    st.dataframe(recs, use_container_width=True)

# ===============================
# 📊 Biểu đồ so sánh mô hình
# ===============================
if not summary_df.empty:
    st.subheader("🏆 Model Performance Chart")
    st.bar_chart(summary_df.set_index("Model")[["Recall@10", "NDCG@10"]])
