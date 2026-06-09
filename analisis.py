# ==========================================================
# ANALISIS DE IMPORTANCIA DE CARACTERISTICAS
# ==========================================================

import pandas as pd

# Random Forest para calcular importancia
from sklearn.ensemble import RandomForestClassifier

# ==========================================================
# CARGAR CSV GENERADO EN EL PASO ANTERIOR
# ==========================================================

df = pd.read_csv("features_normalizadas.csv")

print("\nPrimeras filas:")
print(df.head())

# ==========================================================
# ELIMINAR NOMBRE DEL ARCHIVO
# ==========================================================
#
# El nombre del archivo no describe el sonido.
# Solo es un identificador.
#
# ==========================================================

df = df.drop(columns=["archivo"])

# ==========================================================
# SEPARAR CARACTERISTICAS Y ETIQUETA
# ==========================================================
#
# X = variables acústicas
# y = especie
#
# ==========================================================

X = df.drop(columns=["especie"])

y = df["especie"]

# ==========================================================
# CREAR RANDOM FOREST
# ==========================================================
#
# Usamos pocos árboles porque estamos explorando.
# No estamos buscando máxima precisión.
#
# ==========================================================

modelo = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# ==========================================================
# ENTRENAR
# ==========================================================

modelo.fit(X, y)

# ==========================================================
# EXTRAER IMPORTANCIAS
# ==========================================================

importancias = modelo.feature_importances_

# ==========================================================
# CREAR TABLA
# ==========================================================

ranking = pd.DataFrame({
    "feature": X.columns,
    "importancia": importancias
})

# ==========================================================
# ORDENAR
# ==========================================================

ranking = ranking.sort_values(
    by="importancia",
    ascending=False
)

# ==========================================================
# MOSTRAR RESULTADOS
# ==========================================================

print("\nRANKING DE IMPORTANCIA\n")

print(ranking)

# ==========================================================
# GUARDAR RESULTADOS
# ==========================================================

ranking.to_csv(
    "ranking_importancia.csv",
    index=False
)

print("\nArchivo guardado:")
print("ranking_importancia.csv")