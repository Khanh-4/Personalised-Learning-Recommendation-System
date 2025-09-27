import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Config
NUM_USERS = 500
NUM_ITEMS = 20   # tăng lên 20 items
NUM_SESSIONS = 1000  # tổng số sessions

random.seed(42)
np.random.seed(42)

rows = []

for session_id in range(NUM_SESSIONS):
    learner_id = random.randint(1, NUM_USERS)
    content_type = random.randint(0, NUM_ITEMS - 1)

    # random timestamp trong 3 tháng gần đây
    days_ago = random.randint(0, 90)
    timestamp = datetime.now() - timedelta(days=days_ago, hours=random.randint(0, 23))

    time_spent = np.random.randint(5, 120)
    quiz_score = np.random.randint(0, 101)
    completion_rate = round(np.random.uniform(0, 1), 2)
    attempts_per_quiz = np.random.randint(1, 6)
    learning_outcome = random.choice([0.0, 0.5, 1.0])
    engagement_score = round(
        (0.5 * (quiz_score / 100)) + (0.3 * completion_rate) + (0.2 * (time_spent / 120)), 2
    )

    rows.append([
        learner_id,
        session_id,
        timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        content_type,
        time_spent,
        engagement_score,
        quiz_score,
        completion_rate,
        attempts_per_quiz,
        learning_outcome,
        timestamp.hour,
        timestamp.weekday(),
        timestamp.month
    ])

# Tạo dataframe
columns = [
    "learner_id",
    "session_id",
    "timestamp",
    "content_type",
    "time_spent",
    "engagement_score",
    "quiz_score",
    "completion_rate",
    "attempts_per_quiz",
    "learning_outcome",
    "hour",
    "day_of_week",
    "month",
]

df = pd.DataFrame(rows, columns=columns)

# Xuất ra file CSV
df.to_csv("synthetic_learning_dataset.csv", index=False)
print("✅ Synthetic dataset generated: synthetic_learning_dataset.csv")
print(df.head())
