import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración de rutas relativas
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
OUTPUT_DIR = BASE_DIR / "reports" / "figures"

def ejecutar_eda():
    # Cargar datos
    df = pd.read_csv(DATA_PATH)

    # 1. Resumen de valores faltantes
    print("=== Valores Faltantes ===")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    # 2. Tasa de supervivencia por sexo
    print("\n=== Supervivencia por Sexo ===")
    tabla_sexo = df.groupby('Sex')['Survived'].mean()
    print(tabla_sexo)

    # 3. Generar y guardar gráfico reproducible
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='Sex', y='Survived', hue='Pclass', ci=None)
    plt.title('Tasa de Supervivencia por Sexo y Clase')
    plt.ylabel('Proporción de Supervivencia')
    plt.xlabel('Sexo')

    # Guardar figura
    plt.tight_layout()
    fig_path = OUTPUT_DIR / "supervivencia_sexo_clase.png"
    plt.savefig(fig_path)
    print(f"\nGráfico guardado exitosamente en: {fig_path}")

if __name__ == "__main__":
    ejecutar_eda()