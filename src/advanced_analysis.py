import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Configuración de rutas relativas
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
OUTPUT_DIR = BASE_DIR / "reports" / "figures"

def ejecutar_ingenieria_caracteristicas():
    df = pd.read_csv(DATA_PATH)

    # 1. Imputación de valores faltantes
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # 2. Creación de nuevas características (Feature Engineering)
    # Extraer el título del nombre (Mr, Mrs, Miss, Master, etc.)
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

    # Crear tamaño de familia y si viajaba solo
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # 3. Codificación de variables categóricas
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
    X = pd.get_dummies(df[features], drop_first=True)
    y = df['Survived']

    # 4. División y entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluaciones
    predictions = model.predict(X_test)
    print("=== Métricas con Feature Engineering ===")
    print(f"Exactitud (Accuracy): {accuracy_score(y_test, predictions):.4f}\n")
    print(classification_report(y_test, predictions))

    # 5. Generar y guardar gráfico de Importancia de Variables
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis")
    plt.title("Top 10 Variables más Importantes para la Supervivencia")
    plt.xlabel("Importancia Relativa")
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = OUTPUT_DIR / "importancia_caracteristicas.png"
    plt.savefig(fig_path)
    print(f"Gráfico de importancia guardado en: {fig_path}")

if __name__ == "__main__":
    ejecutar_ingenieria_caracteristicas()