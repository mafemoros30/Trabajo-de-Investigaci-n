import pandas as pd

df = pd.read_csv(r"C:/Users/57316/OneDrive/Escritorio/2025-I/tutorial/RESULTADOS/S001_spatiotemporal_mean2.csv")

# Limpiar columnas
df.columns = df.columns.str.strip()

# Ver columnas que contienen Width
print([c for c in df.columns if "Width" in c])

# Extraer variable correcta
right_step_width = df["Right Step Width(cm)"]

# Estadísticas
mean_val = right_step_width.mean()
min_val  = right_step_width.min()
max_val  = right_step_width.max()

print(f"Right Step Width - Mean: {mean_val:.2f}, Std: {std_val:.2f}, Min: {min_val:.2f}, Max: {max_val:.2f}")
