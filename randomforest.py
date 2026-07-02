# CODIGO 3.1 #

# CALCULO DE IMPORTANCIA DE CARACTERISTICAS
# UTILIZANDO RANDOM FOREST
# mismo codigo que relieff.py pero con RandomForest en lugar de ReliefF

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


RUTA_NORMALIZADO = r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\CARACT_NORMALIZADO.csv"
SALIDA_IMPORTANCIA = r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\IMPORTANCIA_CARACTERISTICAS.csv"
RANDOM_STATE = 42
N_ESTIMATORS = 100

# Primeras columnas que NO son características
# 0 = ruta_audio
# 1 = scientific_name
# 2 = class_name
N_COLUMNAS_META = 3

# Parte 1 #

df = pd.read_csv(RUTA_NORMALIZADO)

print("=" * 60)
print("CARGANDO BASE")
print("=" * 60)

print(f"Registros cargados : {len(df)}")
print(f"Columnas totales   : {len(df.columns)}")

# Parte 2 #

feature_names = list(df.columns[N_COLUMNAS_META:])

print(f"\nCaracteristicas detectadas : {len(feature_names)}")
print("\nLista de caracteristicas:")

for i, nombre in enumerate(feature_names):
    print(f"   [{i}] {nombre}")

# Parte 3 #

X = df.iloc[:, N_COLUMNAS_META:]
# VECTOR y
# Columna 1 = scientific_name
encoder = LabelEncoder()

y = encoder.fit_transform(df.iloc[:, 1])
print(f"\nClases detectadas : {len(encoder.classes_)}")

# Parte 4 #
# RANDOM FOREST

print("\n" + "=" * 60)
print("ENTRENANDO RANDOM FOREST")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X, y)
print("Entrenamiento terminado")

# Parte 5 #

importancias = rf.feature_importances_
df_importancia = pd.DataFrame({
    "indice": range(len(feature_names)),
    "caracteristica": feature_names,
    "importancia_normalizada": importancias
})

df_importancia.to_csv(SALIDA_IMPORTANCIA,index=False)

# Parte 6 #

print("\n" + "=" * 60)
print("TOP CARACTERISTICAS")
print("=" * 60)

for fila in df_importancia.itertuples(index=False):
    print(
        f"[{fila.indice}] "
        f"{fila.caracteristica:<25}"
        f"{fila.importancia_normalizada:.6f}"
    )

# Parte 7 #

print("\n" + "=" * 60)
print("REPORTE FINAL")
print("=" * 60)

print(f"Registros utilizados      : {len(df)}")
print(f"Clases detectadas         : {len(encoder.classes_)}")
print(f"Caracteristicas analizadas: {len(feature_names)}")
print(f"Suma importancias         : {importancias.sum():.6f}")

print("\nArchivo generado:")
print(SALIDA_IMPORTANCIA)

print("\nPROCESO TERMINADO")