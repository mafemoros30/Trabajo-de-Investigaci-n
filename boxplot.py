
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def boxplot_from_csvs(file_paths):
    """
    Genera boxplots para uno o varios archivos CSV.
    
    Parámetros:
        file_paths (list): lista con las rutas de los archivos CSV.
    """
    for path in file_paths:
        # Verifica que el archivo exista
        if not os.path.exists(path):
            print(f"Archivo no encontrado: {path}")
            continue
        
        # Cargar el CSV
        df = pd.read_csv(path)
        outlier_columns = []
        # Filtrar solo columnas numéricas
        numeric_cols = df.select_dtypes(include='number').columns

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 *IQR
            outliers = df[(df[col]< limite_inferior) | (df[col]> limite_superior)]

            if not outliers.empty:
                    print(f"\n Columna: {col}")
                    print(f"   Limite inferior: {limite_inferior:.3f}")
                    print(f"   Limite superior: {limite_superior:.3f}")
                    print("   Valores atipicos encontrados:")
                    print(outliers[[col, 'trial']] if 'trial' in df.columns else outliers[col])

        
        if len(numeric_cols) == 0:
            print(f"No hay columnas numericas en {os.path.basename(path)}")
            continue
        
        # Crear el boxplot
        plt.figure(figsize=(14, 6))
        sns.boxplot(data=df[numeric_cols], orient="v")
        plt.xticks(rotation=45)
        plt.title(f"Boxplot de columnas numericas - {os.path.basename(path)}")
        plt.ylabel("Valor")
        plt.tight_layout()
        plt.show()
        
        print(f"Boxplot generado para: {os.path.basename(path)}\n")


archivos = [
    r"C:/Users/57316/OneDrive/Escritorio/2025-I/tutorial/RESULTADOS/S001_spatiotemporal_mean.csv",

]

boxplot_from_csvs(archivos)

