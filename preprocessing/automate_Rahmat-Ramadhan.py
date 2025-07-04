import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def run_preprocessing(input_path='data.csv', output_path='preprocessing/dataset_preprocessing.csv'):
    df = pd.read_csv(input_path)

    # Drop kolom id (jika ada)
    if 'id' in df.columns:
        df = df.drop(columns='id')

    # Isi missing value
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())

    # Mapping kategori ke numerik
    mapping = {
        'gender': {'Male': 1, 'Female': 0},
        'ever_married': {'Yes': 1, 'No': 0},
        'Residence_type': {'Urban': 1, 'Rural': 0}
    }
    df.replace(mapping, inplace=True)

    # One-hot encoding
    df = pd.get_dummies(df, columns=['work_type', 'smoking_status'])

    # Tangani outlier / skewed numeric
    df['avg_glucose_level'] = np.log1p(df['avg_glucose_level'])

    # Scaling
    scaler = StandardScaler()
    df[['age', 'bmi']] = scaler.fit_transform(df[['age', 'bmi']])

    # Convert bool ke int
    bool_cols = df.select_dtypes('bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Save
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    run_preprocessing()
