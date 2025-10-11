# PERSONALIZED LEARNING RECOMMENDATION SYSTEM

## 1. Giới thiệu

Dự án **Personalized Learning Recommendation System** được phát triển nhằm xây dựng một **hệ thống gợi ý học liệu cá nhân hóa** cho người học dựa trên hành vi học tập, mức độ tương tác, điểm số và tiến độ học tập.

Mục tiêu của hệ thống là:
- Tối ưu hóa trải nghiệm học tập của từng người dùng.
- Gợi ý nội dung học phù hợp với năng lực và sở thích.
- Tăng mức độ hoàn thành khóa học và hiệu quả tiếp thu.

Hệ thống kết hợp nhiều kỹ thuật học máy và mô hình gợi ý khác nhau, bao gồm:
- **Collaborative Filtering (CF)**
- **Content-Based Filtering (CBF)**
- **Matrix Factorization (MF)**
- **Hybrid Recommendation**
- **Contextual Bandit (LinUCB)**

---

## 2. Kiến trúc hệ thống

Hệ thống được thiết kế thành các pipeline chính:

1. **Data Preprocessing**  
   - Đọc dữ liệu từ `dataset/synthetic_learning_dataset.csv`  
   - Xử lý thời gian (`timestamp` → `hour`, `day_of_week`, `month`)  
   - Chuẩn hóa các biến liên tục (`engagement_score`, `quiz_score`, `completion_rate`)  
   - Sinh nhãn `rating` cho người học

2. **Feature Engineering & Model Input**  
   - Tạo các vector biểu diễn người học và khóa học.  
   - Tích hợp thông tin ngữ cảnh (time-based, performance-based).

3. **Model Training & Evaluation**  
   - Huấn luyện và đánh giá nhiều mô hình: CF, MF, Hybrid, Bandit.  
   - Sử dụng các thước đo: Precision@K, Recall@K, RMSE.

4. **Visualization & Reporting**  
   - Xuất kết quả và biểu đồ tại thư mục `charts/` và `results/`.
   - Ví dụ biểu đồ: `charts/model_comparison.png`.

---

## 3. Cấu trúc thư mục

```
Personalised-Learning-Recommendation-System/
│
├── charts/         # Biểu đồ kết quả
├── dataset/        # Dữ liệu đầu vào
├── models/         # File mô hình đã lưu
├── results/        # Kết quả chạy mô hình
├── src/            # Mã nguồn chính (Python)
│   ├── preprocessing.py
│   ├── evaluation.py
│   ├── contextual_bandit.py
│   ├── hybrid_model.py
│   ├── main.py
│   └── utils.py
├── requirements.txt
└── README.md
```

---

## 4. Mô hình sử dụng

| Mô hình | Mô tả | Ưu điểm |
|----------|-------|----------|
| Collaborative Filtering | Dựa vào hành vi người học tương tự | Dễ triển khai, gợi ý tốt với dữ liệu tương tác dày |
| Content-Based Filtering | Dựa trên nội dung học liệu | Phù hợp khi người học mới |
| Matrix Factorization | Phân rã ma trận user-item | Giảm nhiễu, tối ưu tốt |
| Hybrid | Kết hợp CF và MF | Cân bằng giữa chính xác và đa dạng |
| Contextual Bandit (LinUCB) | Gợi ý động theo ngữ cảnh thời gian thực | Cá nhân hóa tức thì |

---

## 5. Kết quả & Đánh giá

Kết quả huấn luyện được lưu trong thư mục `results/`, bao gồm:
- Precision@K, Recall@K, RMSE
- Biểu đồ so sánh mô hình (`charts/model_comparison.png`)
- Các bảng tổng hợp kết quả (`results/summary.csv`)

Hệ thống cho thấy **mô hình Hybrid + LinUCB** cho kết quả tốt nhất trong việc cá nhân hóa học liệu.

---

## 6. Hướng phát triển

- Triển khai API RESTful để tích hợp với hệ thống LMS (Learning Management System).  
- Mở rộng dữ liệu người học thực tế.  
- Ứng dụng Deep Learning (AutoEncoder / Transformers).  

---

## 7. Thông tin nhóm phát triển

**Tác giả:** Khanh  
**GitHub:** [Khanh-4](https://github.com/Khanh-4)  
**Năm thực hiện:** 2025  
**Ngôn ngữ:** Python 3.10  

---
