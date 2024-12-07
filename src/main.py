"""
Title: Naive Bayes Classifier
Deskripsi: Implementasi Naive Bayes Classifier dalam Memprediksi Kelulusan Mahasiswa
Author: M. Farhan Nabil (23051204373), Naufal Andrianto Nugraha (23051204373)
"""

import pandas

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

dataset = pandas.read_csv("data/dataset.csv")

dataset['Target'] = dataset['Target'].map({'Graduate': 1, 'Dropout': 0})

# Handle missing values if any
dataset = dataset.dropna()

# Memisahkan fitur (X) dan label (y)
X = dataset.drop(columns=["Target"])
y = dataset["Target"]

# Fungsi untuk melatih dan mengevaluasi model pada berbagai skenario
def evaluate_naive_bayes(train_size, test_size, scenario_name):
    # Membagi dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=42)
    
    # Melatih model Naive Bayes
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test)
    
    # Evaluasi performa
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"\n=== Scenario {scenario_name} ===")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Menjalankan tiga skenario
evaluate_naive_bayes(train_size=400, test_size=100, scenario_name="1 (400 train, 100 test)")
evaluate_naive_bayes(train_size=250, test_size=250, scenario_name="2 (250 train, 250 test)")
evaluate_naive_bayes(train_size=100, test_size=400, scenario_name="3 (100 train, 400 test)")
