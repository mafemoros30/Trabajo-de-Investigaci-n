import pandas as pd
from pathlib import Path

# =====================================================
# ✅ AQUÍ PONES LA RUTA DE TU ARCHIVO CSV
# =====================================================
csv_path = Path("C:/Users/57316/OneDrive/Escritorio/2025-I/tutorial/RESULTADOS/S001_spatiotemporal_mean2.csv")
# Ejemplo Windows:
# csv_path = Path("C:/Users/Mafer/Documents/S001_spatiotemporal_mean2.csv")

# =====================================================
# 1. Leer el archivo CSV
# =====================================================
df = pd.read_csv(csv_path)


# Limpiar nombres de columnas (quita espacios extra)
df.columns = [c.strip() for c in df.columns]

# =====================================================
# 2. Identificar columnas ID y variables numéricas
# =====================================================
id_cols = ["trial"]  # puedes añadir "patient_id" si quieres

value_cols = [
    c for c in df.columns
    if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])
]

# =====================================================
# 3. Imprimir los 18 valores de cada variable
# =====================================================
for var in value_cols:
    print("\n" + "=" * 100)
    print(f"Variable: {var}")
    print("-" * 100)

    # Mostrar trial + valores
    print(df[["trial", var]].to_string(index=False))

print("\n✅ Listo. Se imprimieron los 18 resultados por variable.")
