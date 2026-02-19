
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
if not os.path.exists("X.npy") or not os.path.exists("y.npy"): 
    exit()
X = np.load("X.npy")
y = np.load("y.npy")
print(f"Loaded X: {X.shape}, y: {y.shape}")
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except Exception as e:
    exit()
model = RandomForestClassifier(n_estimators=100, n_jobs=1, verbose=1)
model.fit(X_train, y_train)
joblib.dump(model, "hand_gesture.joblib")
print("Model saved successfully as hand_gesture.joblib")
