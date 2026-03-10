import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def whisker_por_variable(rutas_archivos, guardar=False, carpeta_salida="whisker_plots"):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    for ruta in rutas_archivos:
        if not os.path.exists(ruta):
            print(f"❌ Archivo no encontrado: {ruta}")
            continue

        print(f"\n{'='*60}")
        print(f"PROCESANDO: {os.path.basename(ruta)}")

        df = pd.read_csv(ruta)

        # Convertir todo a numérico cuando se pueda
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        if guardar:
            os.makedirs(carpeta_salida, exist_ok=True)

        for col in df.columns:
            datos = df[col].dropna()
            if datos.empty:
                continue

            plt.figure(figsize=(6, 5))

            # Whisker plot (boxplot)
            sns.boxplot(y=datos, width=0.35, showfliers=False)

            # Puntos individuales encima
            sns.stripplot(y=datos, jitter=True, alpha=0.5, size=5, color="grey")

            plt.title(f"{col}\n{os.path.basename(ruta)}")
            plt.ylabel(col)
            plt.xticks([])  # no hace falta eje X

            plt.tight_layout()

            if guardar:
                nombre_seguro = "".join(ch if ch.isalnum() or ch in " _-" else "_" for ch in col).strip()
                outpath = os.path.join(
                    carpeta_salida,
                    f"{os.path.basename(ruta).replace('.csv','')}_{nombre_seguro}.png"
                )
                plt.savefig(outpath, dpi=300)
                plt.close()
            else:
                plt.show()


# --- EJECUCIÓN ---
archivos = [
    r"C:\Users\57316\OneDrive\Escritorio\2025-I\tutorial\RESULTADOS\S001_spatiotemporal_mean2.xlsx",
]

whisker_por_variable(archivos, guardar=False)