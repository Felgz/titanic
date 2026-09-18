import pandas as pd
from pathlib import Path

# Definir la ruta relativa al archivo de datos
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"

def analizar_datos():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo en: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    
    print("=== Resumen General del Dataset ===")
    print(df.info())
    print("\n=== Primeras 5 Filas ===")
    print(df.head())
    
    return df

if __name__ == "__main__":
    analizar_datos()