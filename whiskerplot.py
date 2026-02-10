import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def graficar_media_std_con_puntos(rutas_archivos):
    """
    Genera gráficos separados por unidades.
    Muestra:
    1. Puntos individuales (cada ensayo/trial).
    2. La Media (punto central o línea horizontal).
    3. La Desviación Estándar (barra vertical o 'bigote').
    """
    # Estilo limpio para publicación científica
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    for ruta in rutas_archivos:
        if not os.path.exists(ruta):
            print(f"❌ Archivo no encontrado: {ruta}")
            continue
        
        print(f"\n{'='*60}")
        print(f"PROCESANDO: {os.path.basename(ruta)}")
        
        df = pd.read_csv(ruta)
        
        # 1. SEPARACIÓN POR UNIDADES (CRUCIAL para arreglar las líneas planas)
        # Separamos las variables para que cada gráfico tenga una escala lógica.
        grupos = {
            'Distancias (cm)': [c for c in df.columns if '(cm)' in c.lower()],
            'Tiempos (s)': [c for c in df.columns if '(s)' in c.lower() and '(m/s)' not in c.lower()],
            'Velocidad (m/s)': [c for c in df.columns if '(m/s)' in c.lower()],
            'Porcentajes (%)': [c for c in df.columns if '(%)' in c.lower() or 'percent' in c.lower()],
            'Cadencia': [c for c in df.columns if 'cadence' in c.lower()]
        }

        # 2. GENERAR GRÁFICOS
        for nombre_grupo, columnas in grupos.items():
            if not columnas:
                continue 

            print(f"   📈 Generando gráfico para: {nombre_grupo}...")
            
            # Tamaño dinámico según cantidad de variables
            plt.figure(figsize=(max(6, len(columnas) * 1.5), 6))
            
            # A) DIBUJAR LOS PUNTOS (TRIALS)
            # Usamos stripplot para ver la distribución real de los datos
            sns.stripplot(data=df[columnas], jitter=True, alpha=0.4, color="grey", size=5, zorder=0)

            # B) CALCULAR Y DIBUJAR MEDIA Y DESVIACIÓN ESTÁNDAR
            # Iteramos sobre cada columna para dibujar manualmente la media y el error
            for i, col in enumerate(columnas):
                datos_col = df[col].dropna()
                if datos_col.empty:
                    continue
                
                media = datos_col.mean()
                std = datos_col.std()
                
                # Dibujar la Barra de Error (Mean ± SD)
                # x=i pone la barra en la posición correcta de la columna
                plt.errorbar(x=i, y=media, yerr=std, 
                             fmt='none',       # No queremos un punto automático de errorbar
                             ecolor='red',     # Color de la barra de desviación
                             elinewidth=2,     # Grosor de la línea vertical
                             capsize=10,       # Tamaño de los "bigotes" horizontales (Mean+SD y Mean-SD)
                             capthick=2,
                             zorder=5)         # Poner encima de los puntos grises
                
                # Dibujar el punto de la Media (o línea horizontal corta)
                plt.plot(i, media, marker='_', markersize=30, color='red', markeredgewidth=3, zorder=6)

            # Decoración del gráfico
            plt.title(f"{nombre_grupo} (Mean ± SD)\n{os.path.basename(ruta)}")
            plt.xticks(rotation=45, ha='right')
            plt.ylabel(nombre_grupo)
            
            # Ajustar límites Y para que no se vea plano (quita el 0 forzado si es necesario)
            plt.autoscale(enable=True, axis='y', tight=False)
            
            plt.tight_layout()
            plt.show()

# --- EJECUCIÓN ---
archivos = [
    r"C:/Users/57316/OneDrive/Escritorio/2025-I/tutorial/RESULTADOS/S001_spatiotemporal_mean.csv",
]

graficar_media_std_con_puntos(archivos)