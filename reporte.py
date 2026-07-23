    ####################
        # REVISION #
    ####################

# FILTRADO #

import os
import pandas as pd
import soundfile as sf

RUTA_TRAIN = r"C:\Users\manue\Downloads\3ccbe\DATOS\train.csv"
RUTA_FILTRADO= r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\Aves\5aves.csv"
SALIDA_REPORTE = r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\Aves\Reporte1.txt"

train_original = pd.read_csv(RUTA_TRAIN)
train_filtrado = pd.read_csv(RUTA_FILTRADO)

audios_originales = len(train_original)
audios_filtrados = len(train_filtrado)
audios_eliminados = audios_originales - audios_filtrados

especies_originales = train_original["scientific_name"].nunique()
especies_filtradas = train_filtrado["scientific_name"].nunique()
especies_eliminadas = especies_originales - especies_filtradas

duraciones = train_filtrado["duracion_seg"]
audios_0_5 = (duraciones <= 5).sum()
audios_5_10 = ((duraciones > 5) & (duraciones <= 10)).sum()
audios_0_10 = (duraciones <= 10).sum()
promedio_0_5 = duraciones[duraciones <= 5].mean()
promedio_5_10 = duraciones[(duraciones > 5) & (duraciones <= 10)].mean()
promedio_0_10 = duraciones[duraciones <= 10].mean()
duracion_minima = duraciones.min()
duracion_maxima = duraciones.max()

with open(SALIDA_REPORTE, "w", encoding="utf-8") as f:

    f.write("=" * 70 + "\n")
    f.write("REPORTE DE FILTRADO\n")
    f.write("=" * 70 + "\n\n")

    f.write("COMPARACIÓN DE DATASETS\n")
    f.write("-" * 70 + "\n")
    f.write(f"Audios originales : {audios_originales}\n")
    f.write(f"Audios filtrados  : {audios_filtrados}\n")
    f.write(f"Audios eliminados : {audios_eliminados}\n\n")

    f.write(f"Especies originales : {especies_originales}\n")
    f.write(f"Especies filtradas  : {especies_filtradas}\n")
    f.write(f"Especies eliminadas : {especies_eliminadas}\n\n")

    f.write("DURACIÓN DE LOS AUDIOS\n")
    f.write("-" * 70 + "\n")
    f.write(f"Audios de 0 a 5 s  : {audios_0_5}\n")
    f.write(f"Audios de 5 a 10 s : {audios_5_10}\n")
    f.write(f"Audios de 0 a 10 s : {audios_0_10}\n\n")

    f.write(f"Promedio 0-5 s  : {promedio_0_5:.4f} s\n")
    f.write(f"Promedio 5-10 s : {promedio_5_10:.4f} s\n")
    f.write(f"Promedio 0-10 s : {promedio_0_10:.4f} s\n\n")

    f.write(f"Duración mínima : {duracion_minima:.4f} s\n")
    f.write(f"Duración máxima : {duracion_maxima:.4f} s\n")

print("Reporte generado correctamente.")

#################################
# EXTRACCION DE CARACTERISTICAS #
#################################

import pandas as pd

RUTA_ENTRADA = r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\Aves\5aves.csv"
RUTA_SALIDA = r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\Aves\5avescaracteristicas.csv"
RUTA_CSV_LIMPIO = r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\Aves\5avesC_limpio.csv"
SALIDA_REPORTE = r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\Aves\Reporte2.txt"

df_entrada = pd.read_csv(RUTA_ENTRADA)
df_salida = pd.read_csv(RUTA_SALIDA)

audios_entrada = len(df_entrada)
audios_salida = len(df_salida)
audios_eliminados = audios_entrada - audios_salida
especies_entrada = df_entrada["scientific_name"].nunique()
especies_salida = df_salida["scientific_name"].nunique()

conteo = df_salida["scientific_name"].value_counts()
especies_eliminar = conteo[conteo == 1].index
df_limpio = df_salida[~df_salida["scientific_name"].isin(especies_eliminar)].copy()
audios_limpio = len(df_limpio)
especies_limpio = df_limpio["scientific_name"].nunique()
audios_eliminados_limpieza = audios_salida - audios_limpio
especies_eliminadas = especies_salida - especies_limpio


df_limpio.to_csv(RUTA_CSV_LIMPIO, index=False)
with open(SALIDA_REPORTE, "w", encoding="utf-8") as f:
    f.write("="*70 + "\n")
    f.write("REPORTE DE EXTRACCIÓN DE CARACTERÍSTICAS\n")
    f.write("="*70 + "\n\n")

    f.write("EXTRACCIÓN\n")
    f.write("-"*70 + "\n")
    f.write(f"Audios de entrada          : {audios_entrada}\n")
    f.write(f"Audios con características : {audios_salida}\n")
    f.write(f"Audios perdidos            : {audios_eliminados}\n\n")

    f.write("LIMPIEZA\n")
    f.write("-"*70 + "\n")
    f.write(f"Especies eliminadas : {especies_eliminadas}\n")
    f.write(f"Audios eliminados   : {audios_eliminados_limpieza}\n")
    f.write(f"Especies finales    : {especies_limpio}\n")
    f.write(f"Audios finales      : {audios_limpio}\n\n")

    f.write("ESPECIES ELIMINADAS\n")
    f.write("-"*70 + "\n")

    for especie in especies_eliminar:
        f.write(f"{especie}\n")

print("Reporte generado correctamente.")
