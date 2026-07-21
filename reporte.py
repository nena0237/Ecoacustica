# REPORTE DEL DATASET
# Genera un reporte estadístico completamente automático.
# No modifica ningún dato.
# Toda la información se guarda en un único archivo TXT.

import pandas as pd

# CSV ORIGINAL
RUTA_ORIGINAL = r"C:\RUTA\AL\CSV_ORIGINAL.csv"
# CSV FINAL (después del filtrado)
RUTA_FILTRADO = r"C:\RUTA\AL\5aves.csv"
# Archivo de salida
SALIDA_REPORTE = r"C:\RUTA\Reporte_Dataset.txt"
# Criterio utilizado
CRITERIO_FILTRADO = "Duración ≤ 5 segundos"

df_original = pd.read_csv(RUTA_ORIGINAL)
df = pd.read_csv(RUTA_FILTRADO)

numero_audios_originales = len(df_original)
numero_audios_finales = len(df)
numero_audios_descartados = numero_audios_originales - numero_audios_finales

numero_especies = df["scientific_name"].nunique()
duracion_promedio = df["duracion_seg"].mean()
duracion_minima = df["duracion_seg"].min()
duracion_maxima = df["duracion_seg"].max()

distribucion = (
    df["scientific_name"]
    .value_counts()
    .rename_axis("scientific_name")
    .reset_index(name="numero_audios")
)

especie_mas = distribucion.iloc[0]["scientific_name"]
cantidad_mas = distribucion.iloc[0]["numero_audios"]
especie_menos = distribucion.iloc[-1]["scientific_name"]
cantidad_menos = distribucion.iloc[-1]["numero_audios"]

with open(SALIDA_REPORTE, "w", encoding="utf-8") as f:

    f.write("=" * 70 + "\n")
    f.write("REPORTE DEL DATASET\n")
    f.write("=" * 70 + "\n\n")

    f.write("INFORMACIÓN DEL FILTRADO\n")
    f.write("-" * 70 + "\n")
    f.write(f"Número de audios originales : {numero_audios_originales}\n")
    f.write(f"Número de audios descartados: {numero_audios_descartados}\n")
    f.write(f"Número de audios finales    : {numero_audios_finales}\n")
    f.write(f"Criterio de filtrado        : {CRITERIO_FILTRADO}\n\n")

    f.write("ESTADÍSTICAS GENERALES\n")
    f.write("-" * 70 + "\n")
    f.write(f"Número de especies únicas : {numero_especies}\n")
    f.write(f"Número de audios          : {numero_audios_finales}\n")
    f.write(f"Duración promedio (s)     : {duracion_promedio:.4f}\n")
    f.write(f"Duración mínima (s)       : {duracion_minima:.4f}\n")
    f.write(f"Duración máxima (s)       : {duracion_maxima:.4f}\n\n")

    f.write("ESPECIES DESTACADAS\n")
    f.write("-" * 70 + "\n")
    f.write(f"Especie con más audios   : {especie_mas} ({cantidad_mas})\n")
    f.write(f"Especie con menos audios : {especie_menos} ({cantidad_menos})\n\n")

    f.write("DISTRIBUCIÓN POR ESPECIE\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'Especie':<45}{'Número de audios'}\n")
    f.write("-" * 70 + "\n")

    for fila in distribucion.itertuples(index=False):
        f.write(f"{fila.scientific_name:<45}{fila.numero_audios}\n")

    f.write("\n")
    f.write("=" * 70 + "\n")
    f.write("FIN DEL REPORTE\n")
    f.write("=" * 70 + "\n")