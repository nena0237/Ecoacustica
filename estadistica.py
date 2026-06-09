# =====================================================
# ESTADISTICAS DE LAS FEATURES
# =====================================================

import pandas as pd

# -----------------------------------------------------
# CARGAR CSV NORMALIZADO
# -----------------------------------------------------

df = pd.read_csv("features_normalizadas.csv")

# -----------------------------------------------------
# ELIMINAR COLUMNAS NO NUMERICAS
# -----------------------------------------------------

columnas_excluir = ["archivo", "especie"]

features = df.drop(columns=columnas_excluir)

# -----------------------------------------------------
# CALCULAR ESTADISTICAS
# -----------------------------------------------------

estadisticas = pd.DataFrame({
    "media": features.mean(),
    "mediana": features.median(),
    "std": features.std(),
    "min": features.min(),
    "max": features.max()
})

# -----------------------------------------------------
# GUARDAR
# -----------------------------------------------------

estadisticas.to_csv(
    "features_estadisticas.csv",
    index=True
)

print("\nArchivo generado:")
print("features_estadisticas.csv")

print("\nPrimeras filas:")
print(estadisticas.head())