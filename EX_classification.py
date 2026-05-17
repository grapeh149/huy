import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Thư viện cho từng mô hình
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ==========================================
# 0. CHUẨN BỊ DỮ LIỆU CHUNG
# ==========================================
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Chuẩn hóa dữ liệu (Cần thiết cho SVM và KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 1. LOGISTIC REGRESSION
# ==========================================
print("--- LOGISTIC REGRESSION ---")
log_reg = LogisticRegression(max_iter=200, random_state=42)
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)
y_prob_lr = log_reg.predict_proba(X_test) # Xác suất

print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(classification_report(y_test, y_pred_lr, target_names=iris.target_names))


# ==========================================
# 2. DECISION TREE
# ==========================================
print("\n--- DECISION TREE ---")
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")

# Trực quan hóa cây
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=iris.feature_names, 
          class_names=iris.target_names, filled=True, rounded=True)
plt.title("Decision Tree")
plt.show()

# Độ quan trọng của các thuộc tính (Feature importance)
print("Feature importances:")
for name, importance in zip(iris.feature_names, dt.feature_importances_):
    print(f"{name}: {importance:.4f}")


# ==========================================
# 3. RANDOM FOREST
# ==========================================
print("\n--- RANDOM FOREST ---")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

# Sắp xếp và in Feature Importance
importance_df = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(importance_df)

# Vẽ biểu đồ cột độ quan trọng tính năng
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.gca().invert_yaxis() # Đảo ngược trục y để tính năng quan trọng nhất nằm ở trên
plt.xlabel('Importance')
plt.title('Feature Importance (Random Forest)')
plt.show()


# ==========================================
# 4. SUPPORT VECTOR MACHINE (SVM)
# ==========================================
print("\n--- SUPPORT VECTOR MACHINE ---")
# SVM chạy tốt hơn trên dữ liệu đã chuẩn hóa (scaled)
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train_scaled, y_train)

y_pred_svm = svm.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")

# Cấu hình SVM để xuất ra xác suất dự đoán
svm_prob = SVC(probability=True, kernel='rbf', random_state=42)
svm_prob.fit(X_train_scaled, y_train)
y_prob_svm = svm_prob.predict_proba(X_test_scaled)


# ==========================================
# 5. K-NEAREST NEIGHBORS (KNN)
# ==========================================
print("\n--- K-NEAREST NEIGHBORS ---")
# KNN bắt buộc phải sử dụng dữ liệu đã chuẩn hóa (scaled)
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")

# Tìm K tối ưu bằng Cross-Validation
k_range = range(1, 31)
scores = []
for k in k_range:
    knn_cv = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn_cv, X_train_scaled, y_train, cv=5).mean()
    scores.append(score)

# Vẽ biểu đồ tìm K
plt.figure(figsize=(10, 6))
plt.plot(k_range, scores, 'bo-')
plt.xlabel('K')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Finding Optimal K')
plt.show()

optimal_k = list(k_range)[np.argmax(scores)]
print(f"Optimal K: {optimal_k}")


# ==========================================
# 6. NAIVE BAYES
# ==========================================
print("\n--- NAIVE BAYES ---")
gnb = GaussianNB() # Cho các thuộc tính liên tục
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
print(f"GaussianNB Accuracy: {accuracy_score(y_test, y_pred_gnb):.4f}")


# ==========================================
# 7. SO SÁNH TẤT CẢ CÁC MÔ HÌNH
# ==========================================
print("\n--- MODEL COMPARISON ---")
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Decision Tree': DecisionTreeClassifier(max_depth=5),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM': SVC(kernel='rbf'),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB()
}

results = []
for name, model in models.items():
    # Sử dụng dữ liệu đã chuẩn hóa cho SVM và KNN
    if name in ['SVM', 'KNN']:
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    else:
        scores = cross_val_score(model, X_train, y_train, cv=5)
        
    results.append({
        'Model': name,
        'Mean Accuracy': scores.mean(),
        'Std': scores.std()
    })

results_df = pd.DataFrame(results).sort_values('Mean Accuracy', ascending=False)
print(results_df)

# Trực quan hóa kết quả so sánh
plt.figure(figsize=(12, 6))
plt.barh(results_df['Model'], results_df['Mean Accuracy'], color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel('Mean Accuracy')
plt.title('Model Comparison')
plt.show()