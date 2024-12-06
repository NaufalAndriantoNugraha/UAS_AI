"""
Title: Naive Bayes Classifier
Deskripsi: Implementasi Naive Bayes Classifier dalam Memprediksi Kelulusan Mahasiswa
Author: M. Farhan Nabil (23051204373), Naufal Andrianto Nugraha (23051204373)
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay



def main():
    data_path = "data/dataset.csv"
    data_set = pd.read_csv(data_path)

    data_set.head()
    data_set.info()

    # Separate features and target
    X = data_set.drop(columns=['Target'])
    y = data_set['Target']

    # Split data into training (500 samples) and testing (400 samples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=500, test_size=400, random_state=42, stratify=y)

    # Convert categorical columns to numeric if necessary
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)

    # Ensure columns are aligned between train and test sets
    X_train, X_test = X_train.align(X_test, join='left', axis=1)
    X_test.fillna(0, inplace=True)

    # Initialize and train Naive Bayes classifier
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Akurasi: {accuracy}")
    print(f"Report: {report}")
    print("")

    # ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
    # plt.title("Confusion Matrix")
    # plt.show()

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()

    # Define evaluation metrics for the current model
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    values = [
        accuracy * 100,  # Accuracy
        df_report.loc["weighted avg", "precision"] * 100,  # Weighted Precision
        df_report.loc["weighted avg", "recall"] * 100,  # Weighted Recall
        df_report.loc["weighted avg", "f1-score"] * 100,  # Weighted F1-Score
    ]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(metrics, values, marker='o', label="Skenario 1")

    # Customize the chart
    plt.title("Evaluasi Model Naive Bayes", fontsize=16)
    plt.ylabel("Score (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
