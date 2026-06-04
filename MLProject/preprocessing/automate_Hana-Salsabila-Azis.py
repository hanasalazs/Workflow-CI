import os
import requests
import zipfile
import pandas as pd
from io import BytesIO
from sklearn.preprocessing import LabelEncoder

def run_preprocessing_pipeline():
    print("=== Menjalankan Automate Preprocessing Pipeline ===")
    
    raw_folder = "student_por_raw"
    os.makedirs(raw_folder, exist_ok=True)
    
    # Download dan Ekstrak Dataset Mentah
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
    print("Mengunduh dataset dari UCI...")
    response = requests.get(url)
    with zipfile.ZipFile(BytesIO(response.content)) as z:
        z.extractall(raw_folder)
        
    # Data
    df = pd.read_csv(f"{raw_folder}/student-por.csv", sep=';')
    data = df.copy()
    
    # Proses Preprocessing (Encoding)
    print("Melakukan encoding data kategorikal...")
    le = LabelEncoder()
    for col in data.select_dtypes(include='object').columns:
        data[col] = le.fit_transform(data[col])
        
    # Simpan Hasil Bersih
    output_path = "preprocessing/student_performance_preprocessed.csv"
    data.to_csv(output_path, index=False)
    print(f"Sukses! Dataset bersih disimpan di: {output_path}")

if __name__ == '__main__':
    run_preprocessing_pipeline()