# ==========================================================
# NORMALIZAR CARACTERISTICAS ENTRE 0 Y 1
# ==========================================================

import pandas as pd

# ==========================================================
# CARGAR CSV ORIGINAL
# ==========================================================
#
# Extracción de características.
#
# Cada fila = un audio
# Cada columna = una característica acústica
#
# ==========================================================

df = pd.read_csv("features_audios.csv")

# ==========================================================
# COLUMNAS QUE NO DEBEN NORMALIZARSE
# ==========================================================
#
# archivo -> nombre del audio
# especie -> etiqueta BirdCLEF
#
# ==========================================================

columnas_excluir = [
    "archivo",
    "especie"
]

# ==========================================================
# CREAR COPIA
# ==========================================================

df_norm = df.copy()

# ==========================================================
# NORMALIZACION MIN-MAX
# ==========================================================
#
# Formula:
#
# (x - min) / (max - min)
#
# Resultado:
#
# 0 = valor más pequeño encontrado
# 1 = valor más grande encontrado
#
# ==========================================================

for columna in df.columns:

    if columna in columnas_excluir:
        continue

    minimo = df[columna].min()
    maximo = df[columna].max()

    # ------------------------------------------------------
    # Evitar división por cero
    # ------------------------------------------------------

    if minimo == maximo:

        df_norm[columna] = 0

    else:

        df_norm[columna] = (
            (df[columna] - minimo)
            /
            (maximo - minimo)
        )

# ==========================================================
# GUARDAR RESULTADO
# ==========================================================

df_norm.to_csv(
    "features_normalizadas.csv",
    index=False
)

# ==========================================================
# INFORMACION
# ==========================================================

print("\nProceso terminado")

print(
    "\nArchivo generado:"
)

print(
    "features_normalizadas.csv"
)

print(
    "\nPrimeras filas:"
)

print(
    df_norm.head()
)