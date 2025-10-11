# 🎯 Personalized Learning Recommendation System

> 📚 An intelligent recommendation engine that suggests personalized learning materials based on learner engagement, performance, and interaction patterns.

---

## 🚀 Features

✅ Learner behavior tracking  
✅ Data preprocessing & feature engineering  
✅ Multiple recommender models (CF, MF, Hybrid, Bandit)  
✅ Model comparison and evaluation  
✅ Visualization & performance charts  

---

## 🧩 Tech Stack

- **Language:** Python 3.10+  
- **Libraries:** pandas, numpy, scikit-learn, matplotlib  
- **Machine Learning:** Matrix Factorization, Hybrid Recommenders, LinUCB Bandit  
- **Visualization:** Matplotlib, Seaborn  

---

## 🗂️ Project Structure

```
├── charts/         # Visualization outputs
├── dataset/        # Input dataset (.csv)
├── models/         # Saved models
├── results/        # Evaluation results
├── src/            # Main source code
│   ├── preprocessing.py
│   ├── contextual_bandit.py
│   ├── evaluation.py
│   ├── hybrid_model.py
│   ├── main.py
│   └── utils.py
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Khanh-4/Personalised-Learning-Recommendation-System.git
   cd Personalised-Learning-Recommendation-System
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main pipeline**
   ```bash
   python src/main.py
   ```

4. **View results**
   - Charts: `charts/model_comparison.png`
   - Evaluation metrics: `results/summary.csv`

---

## 📊 Example Outputs

| Model | Precision@5 | Recall@5 | RMSE |
|--------|-------------|----------|------|
| CF | 0.72 | 0.69 | 0.85 |
| MF | 0.76 | 0.73 | 0.81 |
| Hybrid | 0.80 | 0.78 | 0.77 |
| LinUCB | **0.83** | **0.81** | **0.74** |

📈 *Hybrid + LinUCB outperformed other baselines.*

![Model Comparison](charts/model_comparison.png)

---

## 🧠 Architecture Overview

```text
Data → Preprocessing → Feature Engineering → Model Training → Evaluation → Visualization
```

---

## 🧩 Future Improvements

- Integrate with LMS platforms via REST API  
- Add Deep Learning recommender models  
- Build a simple frontend dashboard for learners  

---

## 👤 Author

**Khanh**  
📎 [GitHub Profile](https://github.com/Khanh-4)  
📅 2025 | Vietnam  
🧠 Project for Learning Analytics & Recommender Systems

---

⭐ *If you like this project, please give it a star on GitHub!*
