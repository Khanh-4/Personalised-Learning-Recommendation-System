import graphviz
import os

def create_detailed_architecture():
    # Khởi tạo sơ đồ dạng Left → Right
    dot = graphviz.Digraph("Detailed_Architecture", format="png")
    dot.attr(rankdir="LR", dpi="300", splines="spline", concentrate="true")
    dot.attr("node", shape="box", style="rounded,filled", color="#999999", fontname="Arial", fontsize="11")

    # ---------------------
    # 1️⃣ DỮ LIỆU ĐẦU VÀO
    # ---------------------
    dot.node(
        "input",
        "① DỮ LIỆU ĐẦU VÀO\n"
        "- Hồ sơ người học (User Profile)\n"
        "- Học liệu (Learning Materials)\n"
        "- Lịch sử học tập (Activity Log)",
        fillcolor="#FFF2CC"
    )

    # -------------------------------
    # 2️⃣ TIỀN XỬ LÝ & TRÍCH XUẤT ĐẶC TRƯNG
    # -------------------------------
    dot.node(
        "feature",
        "② TIỀN XỬ LÝ & TRÍCH XUẤT ĐẶC TRƯNG\n"
        "- Làm sạch dữ liệu\n"
        "- Mã hóa ID, chuẩn hóa\n"
        "- Sinh vector ngữ nghĩa (Embedding)\n"
        "- Kết hợp đặc trưng KG + MF + Content",
        fillcolor="#FFF2CC"
    )

    # ---------------------
    # 3️⃣ MÔ HÌNH GỢI Ý
    # ---------------------
    dot.node(
        "models",
        "③ MÔ HÌNH GỢI Ý\n\nCác mô-đun gợi ý (Recommendation Models):\n"
        "• CF (Collaborative Filtering) — User/Item-based\n"
        "• NCF (Neural CF) — Mạng nơ-ron MLP\n"
        "• SASRec — Mô hình tuần tự (Sequential)\n"
        "• KG (Knowledge Graph) — Gợi ý dựa tri thức liên kết\n"
        "• Contextual Bandit (LinUCB, E-Greedy) — Học thích ứng theo phản hồi",
        fillcolor="#DAE8FC"
    )

    # ---------------------
    # 4️⃣ GIAO DIỆN HIỂN THỊ GỢI Ý
    # ---------------------
    dot.node(
        "ui",
        "④ GIAO DIỆN HIỂN THỊ GỢI Ý\n"
        "- Ứng dụng Streamlit\n"
        "- Dashboard đánh giá\n"
        "- Báo cáo & biểu đồ (Precision, Recall, MAP...)",
        fillcolor="#D5E8D4"
    )

    # ---------------------
    # MỐI LIÊN HỆ GIỮA CÁC KHỐI
    # ---------------------
    dot.edge("input", "feature", label="Tiền xử lý dữ liệu & sinh đặc trưng", fontsize="10")
    dot.edge("feature", "models", label="Truyền vector đặc trưng đầu vào", fontsize="10")
    dot.edge("models", "ui", label="Hiển thị kết quả gợi ý & biểu đồ", fontsize="10")

    # ---------------------
    # Xuất file PNG
    # ---------------------
    output_path = "../results/charts/system_architecture_detailed"
    dot.render(output_path, format="png", cleanup=True)
    print(f"✅ Sơ đồ kiến trúc chi tiết đã được lưu tại: {output_path}.png")

if __name__ == "__main__":
    create_detailed_architecture()
