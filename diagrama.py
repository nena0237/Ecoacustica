# ¿Cuántas muestras tiene cada especie?
# ¿Las especies tienen aproximadamente el mismo número de muestras?
# ¿Existe un desbalance muy grande?
# ¿Hay especies con muy pocos registros que puedan afectar el entrenamiento?

# GRAFICA DE FRECUENCIA CON LABEL #

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\Aves\segun5avescaracteristicas.csv")
frecuencias = (df["label"].value_counts().sort_index())

# ==============================
# Estadísticas
# ==============================
print(f"Número de especies: {len(frecuencias)}")
print(f"Total de audios: {frecuencias.sum()}")
print(f"Menor frecuencia: {frecuencias.min()}")
print(f"Mayor frecuencia: {frecuencias.max()}")
print(f"Promedio: {frecuencias.mean():.2f}")

# ==============================
# Gráfico
# ==============================
plt.figure(figsize=(18,6))
plt.bar(frecuencias.index.astype(str),frecuencias.values)
plt.title("Frecuencia de muestras por especie (Label)", fontsize=15)
plt.xlabel("Etiqueta (Label)")
plt.ylabel("Número de audios")
plt.xticks(ticks=range(len(frecuencias)),
    labels=frecuencias.index,rotation=90,fontsize=7)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()