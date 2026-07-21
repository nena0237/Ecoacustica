# MODIFICACION #

# CODIGO 3.2 #

# CALCULO DE IMPORTANCIA DE CARACTERISTICAS
# UTILIZANDO RELIEFF
# Este codigo calcula únicamente las importancias entregadas por ReliefF.
# No realiza ningún tipo de normalización ni modificación de los valores.

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from skrebate import ReliefF

RUTA_NORMALIZADO = r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\Filtracion 1\5avescaracteristicas.csv"
SALIDA_IMPORTANCIA = r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\5avesRELIEFF.csv"

N_NEIGHBORS = 100
N_COLUMNAS_META = 4

# ==========================================================
# CARGAR BASE
# ==========================================================

df = pd.read_csv(RUTA_NORMALIZADO)

print("=" * 60)
print("CARGANDO BASE")
print("=" * 60)

print(f"Registros cargados : {len(df)}")
print(f"Columnas totales   : {len(df.columns)}")

feature_names = list(df.columns[N_COLUMNAS_META:])

print(f"\nCaracterísticas detectadas : {len(feature_names)}")
print("\nLista de características:")

for i, nombre in enumerate(feature_names):
    print(f"[{i}] {nombre}")

# ==========================================================
# PREPARAR DATOS
# ==========================================================

X = df.iloc[:, N_COLUMNAS_META:].values

encoder = LabelEncoder()
y = encoder.fit_transform(df.iloc[:, 1])

print(f"\nClases detectadas : {len(encoder.classes_)}")

# ==========================================================
# RELIEFF
# ==========================================================

print("\n" + "=" * 60)
print("CALCULANDO IMPORTANCIAS CON RELIEFF")
print("=" * 60)

relieff = ReliefF(n_neighbors=N_NEIGHBORS)
relieff.fit(X, y)

print("Proceso terminado.")

# ==========================================================
# IMPORTANCIAS ORIGINALES
# ==========================================================

importancias = relieff.feature_importances_

df_importancia = pd.DataFrame({
    "indice": range(len(feature_names)),
    "caracteristica": feature_names,
    "importancia": importancias
})

df_importancia.to_csv(SALIDA_IMPORTANCIA, index=False)

# ==========================================================
# MOSTRAR RESULTADOS
# ==========================================================

print("\n" + "=" * 60)
print("IMPORTANCIAS OBTENIDAS")
print("=" * 60)

for fila in df_importancia.itertuples(index=False):
    print(
        f"[{fila.indice}] "
        f"{fila.caracteristica:<25}"
        f"{fila.importancia:.10f}"
    )

# ==========================================================
# REPORTE FINAL
# ==========================================================

print("\n" + "=" * 60)
print("REPORTE FINAL")
print("=" * 60)

print(f"Registros utilizados       : {len(df)}")
print(f"Clases detectadas          : {len(encoder.classes_)}")
print(f"Características analizadas : {len(feature_names)}")

print(f"\nImportancia mínima : {importancias.min():.10f}")
print(f"Importancia máxima : {importancias.max():.10f}")
print(f"Suma importancias  : {importancias.sum():.10f}")

print("\nArchivo generado:")
print(SALIDA_IMPORTANCIA)

print("\nPROCESO TERMINADO")
