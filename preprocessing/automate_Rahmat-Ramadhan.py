import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def automate_preprocessing(input_path: str, output_path: str):
    # Load data
    df = pd.read_csv(input_path)

    # Drop kolom yang tidak diperlukan
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)

    # Konversi TotalCharges ke numerik (float), tangani error jadi NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop missing values
    df.dropna(inplace=True)

    # Ubah kolom kategorikal ke tipe category
    categorical_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod', 'Churn'
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Label Encoding untuk kolom kategorikal
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Scaling untuk kolom numerik
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Simpan hasil ke file CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\u2705 Preprocessing selesai. Dataset disimpan di: {output_path}")

if __name__ == "__main__":
    input_path = "data.csv"  # ganti sesuai nama file mentahmu
    output_path = "preprocessing/data_automate_processing.csv"
    automate_preprocessing(input_path, output_path)