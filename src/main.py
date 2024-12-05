"""
Title: Naive Bayes Classifier
Deskripsi: Implementasi Naive Bayes Classifier dalam Memprediksi Kelulusan Mahasiswa
Author: M. Farhan Nabil (23051204373), Naufal Andrianto Nugraha (23051204373)
"""

import pandas


def main():
    data_path = "data/dataset.csv"
    data_set = pandas.read_csv(data_path)

    print(data_set)


if __name__ == "__main__":
    main()
