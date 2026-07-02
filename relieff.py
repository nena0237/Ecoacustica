# CODIGO 3.2 #

# CALCULO DE IMPORTANCIA DE CARACTERISTICAS
# UTILIZANDO RELIEFF
# mismo codigo que randomforest.py pero con ReliefF en lugar de RandomForest

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from skrebate import ReliefF

RUTA_NORMALIZADO = r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\CARACT_NORMALIZADO.csv"
SALIDA_IMPORTANCIA = r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\IMPORTANCIA_CARACTERISTICAS_RELIEFF.csv"
N_NEIGHBORS = 100
N_COLUMNAS_META = 3

df = pd.read_csv(RUTA_NORMALIZADO)

print("=" * 60)
print("CARGANDO BASE")
print("=" * 60)

print(f"Registros cargados : {len(df)}")
print(f"Columnas totales   : {len(df.columns)}")

feature_names = list(df.columns[N_COLUMNAS_META:])
print(f"\nCaracteristicas detectadas : {len(feature_names)}")
print("\nLista de caracteristicas:")

for i, nombre in enumerate(feature_names):
    print(f"   [{i}] {nombre}")

X = df.iloc[:, N_COLUMNAS_META:].values

encoder = LabelEncoder()

y = encoder.fit_transform(df.iloc[:, 1])
print(f"\nClases detectadas : {len(encoder.classes_)}")

# Parte de: #
# RELIEFF #

print("\n" + "=" * 60)
print("CALCULANDO IMPORTANCIAS CON RELIEFF")
print("=" * 60)

relieff = ReliefF( n_neighbors=N_NEIGHBORS)
relieff.fit(X, y)
print("Proceso terminado")

importancias = relieff.feature_importances_

importancias = importancias - importancias.min()

if importancias.sum() != 0:
    importancias = importancias / importancias.sum()


df_importancia = pd.DataFrame({
    "indice": range(len(feature_names)),
    "caracteristica": feature_names,
    "importancia_normalizada": importancias
})

df_importancia.to_csv(SALIDA_IMPORTANCIA,index=False)

print("\n" + "=" * 60)
print("TOP CARACTERISTICAS")
print("=" * 60)

for fila in df_importancia.itertuples(index=False):
    print(
        f"[{fila.indice}] "
        f"{fila.caracteristica:<25}"
        f"{fila.importancia_normalizada:.6f}"
    )

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