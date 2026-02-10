#corrections for step lenght and time
import os
import numpy as np
import pandas as pd
from gait_events import gait_events_HC_JA
from summary_utils import ensure_dir
import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

#datos = pd.read_excel('C:/Users/57316/OneDrive/Escritorio/2025-I/TRABAJO DE GRADO I/DATA SETS/S001/S001_organizado/S001_G01_D01_B01_T01.xlsx', index_col=0,
#                      engine = 'openpyxl')

# Cargar el Excel
# print(datos.head())



# function to calculate the eucledian distance in mm 
def compute_distance(x1, y1, x2, y2, conversion=10): 
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / conversion


# fuction to calculate stance time considering 4 different cases 
def compute_stance_time_detailed(time, heel_strikes, toe_offs):
    stance_times = []
    n_hs = len(heel_strikes) # número total de eventos heel strike detectados.
    n_to = len(toe_offs) # número total de eventos toe off detectados.
    if n_hs == 0 or n_to == 0:
        return np.array([]) # Si no hay ninguno de los dos tipos de evento, no se puede calcular stance time → devuelve un array vacío.
    
    if n_hs == n_to: # El último HS es posterior al último TO
        if heel_strikes[-1] > toe_offs[-1]:
            for j in range(n_hs - 1):
                #stance = abs(time[toe_offs[j]] - time[heel_strikes[j]])
                stance = abs(time[toe_offs[j + 1]] - time[heel_strikes[j]])
                stance_times.append(stance)
        elif heel_strikes[-1] < toe_offs[-1]:
            for j in range(n_to - 1):
                #stance = abs(time[toe_offs[j]] - time[heel_strikes[j]])
                stance = abs(time[toe_offs[j]] - time[heel_strikes[j + 1]])
                stance_times.append(stance)
    elif n_hs < n_to:
            for j in range(n_hs):
                if j + 1 < len(toe_offs):
                    #stance = abs(time[toe_offs[j]] - time[heel_strikes[j]])
                    stance = abs(time[toe_offs[j + 1]] - time[heel_strikes[j]])
                    stance_times.append(stance)
    elif n_hs > n_to:
        if heel_strikes[-1] > toe_offs[-1]:
            for j in range(n_to):
                #stance = abs(time[toe_offs[j]] - time[heel_strikes[j]])
                stance = abs(time[toe_offs[j]] - time[heel_strikes[j]])
                stance_times.append(stance)
        elif heel_strikes[-1] < toe_offs[-1]:
            for j in range(n_to):
                if j + 1 < len(heel_strikes):
                    #stance = abs(time[toe_offs[j]] - time[heel_strikes[j]])
                    stance = abs(time[toe_offs[j]] - time[heel_strikes[j + 1]])
                    stance_times.append(stance)
    return np.array(stance_times)

# function to calculate swing time 
#swing phase starts with toe off and ends with first contact of the same foot
def compute_swing_time(stride_times, stance_times):
    """
    Computes swing time as the difference between stride time and stance time.
   """
    stride_times = np.asarray(stride_times)
    stance_times = np.asarray(stance_times)    
    # Usar el número mínimo de elementos para evitar errores en la resta
    n = min(len(stride_times), len(stance_times)) # Garantiza que si una lista es más corta, no intente restar elementos que no existen.
    if n < 1:
        return np.array([])    
    # Calcular swing time como la diferencia entre stride y stance time
    swing_times = stride_times[:n] - stance_times[:n]
    return swing_times

 
def compute_L_step_width_and_length(df, heel_strike_R, heel_strike_L, toe_off_L):

    # Trayectorias
    RX = df["Noraxon MyoMotion-Trajectories-Heel RT-x (mm)"].values
    RY = df["Noraxon MyoMotion-Trajectories-Heel RT-y (mm)"].values
    LX = df["Noraxon MyoMotion-Trajectories-Heel LT-x (mm)"].values
    LY = df["Noraxon MyoMotion-Trajectories-Heel LT-y (mm)"].values
    time = df["time"].values

    # Outputs
    right_stride_length = []
    right_stride_time   = []
    left_step_length    = []
    left_step_width     = []
    left_step_time      = []

    # Loop por cada stride derecho
    for i in range(len(heel_strike_R) - 1):

        R0 = heel_strike_R[i]
        R1 = heel_strike_R[i + 1]

        # Dirección del stride derecho
        v = np.array([RX[R1] - RX[R0], RY[R1] - RY[R0]])
        norm = np.linalg.norm(v)
        if norm == 0:
            continue
        D = v / norm

        # Stride length/time (derecho)
        stride_len = norm / 10  # mm → cm
        stride_time = time[R1] - time[R0]

        right_stride_length.append(stride_len)
        right_stride_time.append(stride_time)

        # Buscar HS izquierdo dentro del stride
        candidates = [L for L in heel_strike_L if R0 < L <= R1]
        if not candidates:
            continue

        L0 = candidates[0]

        # Vector desde pie derecho al izquierdo
        vL = np.array([LX[L0] - RX[R0], LY[L0] - RY[R0]])

        # Step length (proyección)
        step_len = np.dot(vL, D) / 10  # mm → cm
        left_step_length.append(step_len)

        # Step width (perpendicular)
        lateral_vec = vL - np.dot(vL, D) * D
        step_wid = np.linalg.norm(lateral_vec) / 10
        left_step_width.append(step_wid)

        # Step time
        left_step_time.append(time[L0] - time[R0])

    # Stance time izquierdo
    left_stance_time = compute_stance_time_detailed(time, heel_strike_L, toe_off_L)

    return (np.array(right_stride_length),
            np.array(left_step_width),
            np.array(left_step_length),
            np.array(right_stride_time),
            np.array(left_step_time),
            left_stance_time)

# Cálculos para el lado derecho (usando eventos del pie izquierdo y toe-off derecho)
def compute_R_step_width_and_length(df, heel_strike_R, heel_strike_L, toe_off_R):

    RX = df["Noraxon MyoMotion-Trajectories-Heel RT-x (mm)"].values
    RY = df["Noraxon MyoMotion-Trajectories-Heel RT-y (mm)"].values
    LX = df["Noraxon MyoMotion-Trajectories-Heel LT-x (mm)"].values
    LY = df["Noraxon MyoMotion-Trajectories-Heel LT-y (mm)"].values
    time = df["time"].values

    left_stride_length = []
    left_stride_time   = []
    right_step_length  = []
    right_step_width   = []
    right_step_time    = []

    # Loop por cada stride izquierdo
    for i in range(len(heel_strike_L) - 1):

        L0 = heel_strike_L[i]
        L1 = heel_strike_L[i + 1]

        # Dirección del stride izquierdo
        v = np.array([LX[L1] - LX[L0], LY[L1] - LY[L0]])
        norm = np.linalg.norm(v)
        if norm == 0:
            continue
        D = v / norm

        # Stride length/time (izquierdo)
        stride_len = norm / 10
        stride_time = time[L1] - time[L0]

        left_stride_length.append(stride_len)
        left_stride_time.append(stride_time)

        # Buscar HS derecho dentro del stride
        candidates = [R for R in heel_strike_R if L0 < R <= L1]
        if not candidates:
            continue

        R0 = candidates[0]

        # Vector desde pie izquierdo al derecho
        vR = np.array([RX[R0] - LX[L0], RY[R0] - LY[L0]])

        # Step length
        step_len = np.dot(vR, D) / 10
        right_step_length.append(step_len)

        # Step width
        lateral_vec = vR - np.dot(vR, D) * D
        step_wid = np.linalg.norm(lateral_vec) / 10
        right_step_width.append(step_wid)

        # Step time
        right_step_time.append(time[R0] - time[L0])

    # Stance time derecho
    right_stance_time = compute_stance_time_detailed(time, heel_strike_R, toe_off_R)

    return (np.array(left_stride_length),
            np.array(right_step_width),
            np.array(right_step_length),
            np.array(left_stride_time),
            np.array(right_step_time),
            right_stance_time)



# Cálculo de la cadencia: pasos por minuto
def compute_cadence(df, heel_strike_R, heel_strike_L):
    time = df["time"].values #Convierte la columna time en un array NumPy para poder acceder a valores por posición.
    total_steps = len(heel_strike_R) + len(heel_strike_L) # Cada HS se considera un paso.
    if len(time) < 2:# Si hay menos de 2 mediciones de tiempo, no se puede calcular la duración
        return np.nan
    total_time_sec = time[-1] - time[0]
    if total_time_sec <= 0: #Si la duración es cero o negativa, devuelve NaN (datos inválidos).
        return np.nan
    return (total_steps / total_time_sec) * 60

# Cálculo de los tiempos de soporte a partir de los datos de contacto
def compute_support_times(df, sampling_rate, heel_strike_R, heel_strike_L, toe_off_R, toe_off_L):
    right_contact = df["Contact RT"].values
    left_contact = df["Contact LT"].values
    total_contacts = right_contact + left_contact

    double_support_time = []
    single_support_time = []
    pct_double = []
    pct_single = []

    for n in range(min(len(heel_strike_R), len(heel_strike_L), len(toe_off_R), len(toe_off_L))-1):
        start = heel_strike_R[n]
        stop = heel_strike_R[n + 1]
        cycle = total_contacts[start:stop + 1]
        stride_duration = (stop - start) / sampling_rate
        double_rows = np.sum(cycle == 2000)
        double_sec = double_rows / sampling_rate
        double_support_time.append(double_sec)
        single_sec = stride_duration - double_sec
        single_support_time.append(single_sec)
        pct_double.append((double_sec / stride_duration) * 100 if stride_duration > 0 else np.nan)
        pct_single.append((single_sec / stride_duration) * 100 if stride_duration > 0 else np.nan)

    return (np.array(double_support_time), np.array(single_support_time),
            np.array(pct_double), np.array(pct_single))

# Cálculo de la distancia recorrida y la velocidad promedio usando la trayectoria del pelvis
def compute_distance_traveled_and_speed(df):
    pelvis_x = df["Noraxon MyoMotion-Trajectories-Pelvis-x (mm)"].values  # Ajusta si es necesario
    pelvis_y =  df["Noraxon MyoMotion-Trajectories-Pelvis-y (mm)"].values
    diff_x = np.diff(pelvis_x)
    diff_y = np.diff(pelvis_y)
    distance_traveled = np.sum(np.sqrt(diff_x ** 2 + diff_y ** 2)) / 1000  # mm a metros
    total_time = df.iloc[-1, 0]  # Se asume que la primera columna es tiempo (s)
    average_speed = distance_traveled / total_time if total_time > 0 else np.nan
    return distance_traveled, average_speed

# Cálculo de la velocidad de zancada (m/s) a partir de la longitud (cm) y el tiempo (s)
def compute_stride_speed(stride_length, stride_time):
    stride_time_nonzero = np.where(stride_time == 0, np.nan, stride_time)
    return (stride_length / stride_time_nonzero) / 100

# Function to define spatiotemporal variables
def compute_spatiotemporal_variables(df, heel_strike_R, heel_strike_L, toe_off_R, toe_off_L, sampling_rate):
    # Cálculos para el lado izquierdo (usando eventos del pie derecho y toe-off izquierdo)
    (right_stride_length, left_step_width, left_step_length,
     right_stride_time, left_step_time,  left_stance_time) = compute_L_step_width_and_length(
         df, heel_strike_R, heel_strike_L, toe_off_L
     )
    
    # Cálculos para el lado derecho (usando eventos del pie izquierdo y toe-off derecho)
    (left_stride_length, right_step_width, right_step_length,
     left_stride_time, right_step_time, right_stance_time) = compute_R_step_width_and_length(
         df, heel_strike_R, heel_strike_L, toe_off_R
     )
    
    cadence = compute_cadence(df, heel_strike_R, heel_strike_L)
    (double_support_time, single_support_time, pct_double, pct_single) = compute_support_times(df, sampling_rate, heel_strike_R, heel_strike_L, toe_off_R, toe_off_L)
    
    right_swing_time = compute_swing_time(right_stride_time, right_stance_time)
    left_swing_time = compute_swing_time(left_stride_time, left_stance_time)
    
    # Evitar división por cero en cálculos de porcentajes
    left_stride_time_nonzero = np.where(left_stride_time == 0, np.nan, left_stride_time)
    right_stride_time_nonzero = np.where(right_stride_time == 0, np.nan, right_stride_time)

    min_len_left = min(len(left_stance_time), len(left_swing_time), len(left_stride_time_nonzero))
    left_stance_time = np.asarray(left_stance_time)[:min_len_left]
    left_swing_time  = np.asarray(left_swing_time)[:min_len_left]
    left_stride_time_nonzero = np.asarray(left_stride_time_nonzero)[:min_len_left]
    left_pct_stance = (left_stance_time / left_stride_time_nonzero) * 100
    left_pct_swing = (left_swing_time / left_stride_time_nonzero) * 100

    min_len_right = min(len(right_stance_time), len(right_swing_time), len(right_stride_time_nonzero))
    right_stance_time = np.asarray(right_stance_time)[:min_len_right]
    right_swing_time  = np.asarray(right_swing_time)[:min_len_right]
    right_stride_time_nonzero = np.asarray(right_stride_time_nonzero)[:min_len_right]
    right_pct_stance = (right_stance_time / right_stride_time_nonzero) * 100
    right_pct_swing = (right_swing_time / right_stride_time_nonzero) * 100

    distance_traveled, average_speed = compute_distance_traveled_and_speed(df)
    left_stride_speed = compute_stride_speed(left_stride_length, left_stride_time)
    right_stride_speed = compute_stride_speed(right_stride_length, right_stride_time)

    # Combinar todos los resultados en un DataFrame. Se rellenan los arrays más cortos con NaN.
    arrays = [right_stride_length, left_step_width, left_step_length, right_stride_time,
              left_step_time, left_swing_time, left_stance_time, left_stride_length,
              right_step_width, right_step_length, right_step_time, right_stance_time,
              right_swing_time, double_support_time, single_support_time, pct_double, pct_single,
              left_pct_stance, left_pct_swing, right_pct_stance, right_pct_swing,
              left_stride_speed, right_stride_speed]
    
    max_length = max(len(arr) for arr in arrays)
   
    def pad_array(arr):
        return np.pad(arr, (0, max_length - len(arr)), constant_values=np.nan)
    
    spatiotemporal_data = {
        "Right Step Width (cm)": pad_array(right_step_width),
        "Left Step Width (cm)": pad_array(left_step_width),
        "Right Stride Length (cm)": pad_array(right_stride_length),
        "Left Stride Length (cm)": pad_array(left_stride_length),
        "Right Step Length (cm)": pad_array(right_step_length),
        "Left Step Length (cm)": pad_array(left_step_length),
        "Right Stride Time (s)": pad_array(right_stride_time),
        "Left Stride Time (s)": pad_array(left_stride_time),
        "Right Step Time (s)": pad_array(right_step_time),
        "Left Step Time (s)": pad_array(left_step_time),
        "Right Stance Time (s)": pad_array(right_stance_time),
        "Right Swing Time (s)": pad_array(right_swing_time),
        "Left Swing Time (s)": pad_array(left_swing_time),
        "Left Stance Time (s)": pad_array(left_stance_time),
        "Double Support Time (s)": pad_array(double_support_time),
        "Single Support Time (s)": pad_array(single_support_time),
        "Percentage Double Support (%)": pad_array(pct_double),
        "Percentage Single Support (%)": pad_array(pct_single),
        "Left Percentage Stance (%)": pad_array(left_pct_stance),
        "Left Percentage Swing (%)": pad_array(left_pct_swing),
        "Right Percentage Stance (%)": pad_array(right_pct_stance),
        "Right Percentage Swing (%)": pad_array(right_pct_swing),
        "Left Stride Speed (m/s)": pad_array(left_stride_speed),
        "Right Stride Speed (m/s)": pad_array(right_stride_speed),
        "Cadence (steps/min)": np.full(max_length, cadence)
    }
    
    spatiotemporal_df = pd.DataFrame(spatiotemporal_data)
    
    mean_spatiotemporal = {
        "Right Step Width (cm)": np.nanmean(right_step_width),
        "Left Step Width (cm)": np.nanmean(left_step_width),
        "Right Stride Length (cm)": np.nanmean(right_stride_length),
        "Left Stride Length (cm)": np.nanmean(left_stride_length),
        "Right Step Length (cm)": np.nanmean(right_step_length),
        "Left Step Length (cm)": np.nanmean(left_step_length),
        "Right Stride Time (s)": np.nanmean(right_stride_time),
        "Left Stride Time (s)": np.nanmean(left_stride_time),
        "Right Step Time (s)": np.nanmean(right_step_time),
        "Left Step Time (s)": np.nanmean(left_step_time),
        "Right Stance Time (s)": np.nanmean(right_stance_time),
        "Right Swing Time (s)": np.nanmean(right_swing_time),
        "Left Swing Time (s)": np.nanmean(left_swing_time),
        "Left Stance Time (s)": np.nanmean(left_stance_time),
        "Double Support Time (s)": np.nanmean(double_support_time),
        "Single Support Time (s)": np.nanmean(single_support_time),
        "Percentage Double Support (%)": np.nanmean(pct_double),
        "Percentage Single Support (%)": np.nanmean(pct_single),
        "Left Percentage Stance (%)": np.nanmean(left_pct_stance),
        "Left Percentage Swing (%)": np.nanmean(left_pct_swing),
        "Right Percentage Stance (%)": np.nanmean(right_pct_stance),
        "Right Percentage Swing (%)": np.nanmean(right_pct_swing),
        "Left Stride Speed (m/s)": np.nanmean(left_stride_speed),
        "Right Stride Speed (m/s)": np.nanmean(right_stride_speed)  
    }
    mean_df=pd.DataFrame([mean_spatiotemporal])

    std_spatiotemporal = {
        "Right Step Width (cm)": np.nanstd(right_step_width),
        "Left Step Width (cm)": np.nanstd(left_step_width),
        "Right Stride Length (cm)": np.nanstd(right_stride_length),
        "Left Stride Length (cm)": np.nanstd(left_stride_length),
        "Right Step Length (cm)": np.nanstd(right_step_length),
        "Left Step Length (cm)": np.nanstd(left_step_length),
        "Right Stride Time (s)": np.nanstd(right_stride_time),
        "Left Stride Time (s)": np.nanstd(left_stride_time),
        "Right Step Time (s)": np.nanstd(right_step_time),
        "Left Step Time (s)": np.nanstd(left_step_time),
        "Right Stance Time (s)": np.nanstd(right_stance_time),
        "Right Swing Time (s)": np.nanstd(right_swing_time),
        "Left Swing Time (s)": np.nanstd(left_swing_time),
        "Left Stance Time (s)": np.nanstd(left_stance_time),
        "Double Support Time (s)": np.nanstd(double_support_time),
        "Single Support Time (s)": np.nanstd(single_support_time),
        "Percentage Double Support (%)": np.nanstd(pct_double),
        "Percentage Single Support (%)": np.nanstd(pct_single),
        "Left Percentage Stance (%)": np.nanstd(left_pct_stance),
        "Left Percentage Swing (%)": np.nanstd(left_pct_swing),
        "Right Percentage Stance (%)": np.nanstd(right_pct_stance),
        "Right Percentage Swing (%)": np.nanstd(right_pct_swing),
        "Left Stride Speed (m/s)": np.nanstd(left_stride_speed),
        "Right Stride Speed (m/s)": np.nanstd(right_stride_speed)
    }
    std_df=pd.DataFrame([std_spatiotemporal])
    
    return spatiotemporal_df, mean_df, std_df

# function to calculate the gait events and spatiotemporal variables
def process_spatiotemporal_for_patient(patient_df,
                                       patient_id,
                                       output_folder,
                                       sampling_rate,
                                       verbose=False):
    """
    For each trial in patient_df:
      1) detect gait events
      2) compute spatiotemporal variables (mean & std)
    Then:
      • aggregate all trial means into one DataFrame
      • aggregate all trial stds  into one DataFrame
      • save both as CSVs in output_folder
    """
    ensure_dir(output_folder)
    mean_results = []
    std_results  = []

    # get distinct trials
    trials = patient_df[['day','block','trial']].drop_duplicates()

    for _, t in trials.iterrows():
        d, b, tr = t['day'], t['block'], t['trial']
        trial_name = f"{patient_id}_{d}_{b}_{tr}"
        df_trial = patient_df[
            (patient_df['day']==d) &
            (patient_df['block']==b) &
            (patient_df['trial']==tr)
        ]

        # 1) Gait‐event detection
        try:
            if verbose: print(f"Detecting gait events for {trial_name}...")
            hs_R, hs_L, to_R, to_L = gait_events_HC_JA(df_trial)
        except Exception as e:
            print(f"[ERROR] gait-event detection failed on {trial_name}: {e}")
            continue

        if not hs_R or not to_R:
            if verbose: print(f"[WARN] Skipping {trial_name}: insufficient events")
            continue

        # 2) Spatiotemporal computation
        try:
            if verbose: print(f"Computing spatiotemporal vars for {trial_name}...")
            _, mean_df, std_df = compute_spatiotemporal_variables(
                df_trial, hs_R, hs_L, to_R, to_L, sampling_rate
            )
            if mean_df.empty or std_df.empty:
                if verbose: print(f"[WARN] Skipping {trial_name}: empty output")
                continue
        except Exception as e:
            print(f"[ERROR] spatiotemporal computation failed on {trial_name}: {e}")
            continue

        # annotate both mean and std tables
        for df_tab, collector in ((mean_df, mean_results),
                                  (std_df,  std_results)):
            df_copy = df_tab.copy()
            df_copy['patient_id'] = patient_id
            df_copy['trial']      = trial_name
            collector.append(df_copy)

    # Combine and save mean
    if mean_results:
        final_mean = pd.concat(mean_results, ignore_index=True)
        mean_path = os.path.join(output_folder, f"{patient_id}_spatiotemporal_mean2.csv")
        if os.path.exists(mean_path): os.remove(mean_path)
        final_mean.to_csv(mean_path, index=False)
        if verbose: print(f"[INFO] Saved mean summary: {mean_path}")

    # Combine and save std
    if std_results:
        final_std = pd.concat(std_results, ignore_index=True)
        std_path = os.path.join(output_folder, f"{patient_id}_spatiotemporal_std2.csv")
        if os.path.exists(std_path): os.remove(std_path)
        final_std.to_csv(std_path, index=False)
        if verbose: print(f"[INFO] Saved std summary: {std_path}")

    if not mean_results and not std_results and verbose:
        print(f"[WARN] No spatiotemporal data for patient {patient_id}")

def concatenar_datos_espaciotemporales():
    # Ruta de la carpeta con tus archivos
    carpeta = Path(r"C:/Users/57316/OneDrive/Escritorio/2025-I/TRABAJO DE GRADO I/DATA SETS/S001/S001")  # ej.: ...\DATA SETS\S001\S001_organizado

    # Regex para nombres tipo S001_G01_D01_B01_T01.csv
    patron = re.compile(r"^(S\d+)_G(\d+)_D(\d+)_B(\d+)_T(\d+)\.csv$", re.IGNORECASE)

    # Si quieres solo la carpeta actual usa: archivos = carpeta.glob("*.csv")
    archivos = carpeta.rglob("*.csv")  # incluye subcarpetas

    dfs = []
    no_match = []

    for p in archivos:
        m = patron.match(p.name)
        if not m:
            no_match.append(p.name)
            continue

        paciente, grupo, dia, bloque, sesion = m.groups()

        # Si tus CSV usan ';' cambia sep=';'
        df = pd.read_csv(p, encoding="utf-8", sep=",")  

        # Añadir columnas con metadatos
        df["day"] = dia
        df["block"] = bloque
        df["trial"] = sesion


        dfs.append(df)

    # Combinar todo
    if not dfs:
        raise RuntimeError("No se encontraron CSV válidos con el patrón esperado.")

    df_total = pd.concat(dfs, ignore_index=True)

    # Guardar combinados
    salida_csv = carpeta / "datos_combinados.csv"
    df_total.to_csv(salida_csv, index=False, encoding="utf-8")
    print(f"✅ CSV combinado guardado en: {salida_csv}")

    # (Opcional) guardar también en Parquet (más eficiente)
    # df_total.to_parquet(carpeta / "datos_combinados.parquet", index=False)

    # Reporte de archivos que no cumplieron el patrón
    if no_match:
        print("⚠️ Archivos ignorados por no cumplir el patrón:")
        for n in no_match:
            print(" -", n)
    process_spatiotemporal_for_patient(df_total,"S001","C:/Users/57316/OneDrive/Escritorio/2025-I/tutorial/RESULTADOS",100,True)

concatenar_datos_espaciotemporales()
