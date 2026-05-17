import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# =====================================================================
# Exercise 3.1: Binary Classification (Breast Cancer Dataset)
# =====================================================================
print("==================================================")
print("EXERCISE 3.1: BREAST CANCER BINARY CLASSIFICATION")
print("==================================================")
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_c, y_c = cancer.data, cancer.target

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_c, y_c, test_size=0.2, random_state=42
)

# Chuẩn hóa dữ liệu cho SVM và Logistic Regression
scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

# 1. Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_c_scaled, y_train_c)
y_pred_lr = lr_model.predict(X_test_c_scaled)
print("\n--- Logistic Regression Report ---")
print(f"Accuracy: {accuracy_score(y_test_c, y_pred_lr):.4f}")
print(classification_report(y_test_c, y_pred_lr, target_names=cancer.target_names))

# 2. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_c, y_train_c) # Cây không nhất thiết phải dùng dữ liệu chuẩn hóa
y_pred_rf = rf_model.predict(X_test_c)
print("\n--- Random Forest Report ---")
print(f"Accuracy: {accuracy_score(y_test_c, y_pred_rf):.4f}")
print(classification_report(y_test_c, y_pred_rf, target_names=cancer.target_names))

# 3. SVM
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_c_scaled, y_train_c)
y_pred_svm = svm_model.predict(X_test_c_scaled)
print("\n--- SVM Report ---")
print(f"Accuracy: {accuracy_score(y_test_c, y_pred_svm):.4f}")
print(classification_report(y_test_c, y_pred_svm, target_names=cancer.target_names))


# =====================================================================
# Exercise 3.2: Multi-class Classification (Iris Overfitting Study)
# =====================================================================
print("\n==================================================")
print("EXERCISE 3.2: IRIS MULTI-CLASS & OVERFITTING")
print("==================================================")
from sklearn.datasets import load_iris

iris = load_iris()
X_i, y_i = iris.data, iris.target

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_i, y_i, test_size=0.3, random_state=42 # Tăng test_size để dễ quan sát phân rã dữ liệu
)

depths = [3, 4, 5]
for d in depths:
    dt_clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt_clf.fit(X_train_i, y_train_i)
    
    # Dự đoán trên tập Train và Test để kiểm tra Overfitting
    train_acc = accuracy_score(y_train_i, dt_clf.predict(X_train_i))
    test_acc = accuracy_score(y_test_i, dt_clf.predict(X_test_i))
    
    print(f"\n[Tree Depth = {d}]")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy : {test_acc:.4f}")
    
    # Trực quan hóa cấu trúc cây tương ứng
    plt.figure(figsize=(12, 6))
    plot_tree(dt_clf, feature_names=iris.feature_names, 
              class_names=iris.target_names, filled=True, rounded=True)
    plt.title(f"Decision Tree (max_depth={d})")
    plt.show()

print("\n Thảo luận về Hiện tượng Overfitting:")
print("Khi max_depth tăng, mô hình có xu hướng học quá chi tiết cả những điểm nhiễu trong tập Train.")
print("Nếu độ chính xác tập Train đạt 1.0000 nhưng tập Test sụt giảm hoặc đi ngang, đó là dấu hiệu Overfitting.")


# =====================================================================
# Exercise 3.3: Titanic Survival Complete Pipeline
# =====================================================================
print("\n==================================================")
print("EXERCISE 3.3: TITANIC SURVIVAL PIPELINE")
print("==================================================")

# Khởi tạo dữ liệu Titanic giả lập khớp cấu trúc bài toán thực tế
np.random.seed(42)
n_samples = 500
titanic_data = sns.load_dataset('titanic')

X_t = titanic_data.drop('Survived', axis=1)
y_t = titanic_data['Survived']

# 1. Thiết lập các bước Tiền xử lý dữ liệu (Preprocessing)
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Điền khuyết bằng trung vị
    ('scaler', StandardScaler())                  # Chuẩn hóa phân phối
])

categorical_features = ['Sex', 'Embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Điền khuyết bằng mode
    ('onehot', OneHotEncoder(handle_unknown='ignore'))    # Chuyển đổi nhãn chữ thành số
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 2. Định nghĩa danh sách các mô hình chạy so sánh
titanic_models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42)
}

# 3. Huấn luyện, áp dụng Cross-Validation lồng trong Pipeline
print("\nSo sánh độ chính xác qua Cross-Validation (5-Fold):")
best_model_name = None
best_score = 0

for name, model in titanic_models.items():
    # Tạo đóng gói toàn trình từ xử lý thô đến phân loại mô hình
    clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', model)])
    
    cv_scores = cross_val_score(clf_pipeline, X_t, y_t, cv=5, scoring='accuracy')
    mean_score = cv_scores.mean()
    print(f"- {name}: Mean Accuracy = {mean_score:.4f} (+/- {cv_scores.std():.4f})")
    
    if mean_score > best_score:
        best_score = mean_score
        best_model_name = name

print(f"\n--> Mô hình được lựa chọn tốt nhất dựa vào Cross-Validation: {best_model_name}")

# 4. Huấn luyện lại mô hình tốt nhất và làm bài kiểm thử (Dự đoán tập test)
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_t, y_t, test_size=0.2, random_state=42)

final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', titanic_models[best_model_name])])

final_pipeline.fit(X_train_t, y_train_t)
final_predictions = final_pipeline.predict(X_test_t)

print(f"\nKết quả kiểm tra cuối cùng trên tập Test độc lập của {best_model_name}:")
print(f"Accuracy Score: {accuracy_score(y_test_t, final_predictions):.4f}")