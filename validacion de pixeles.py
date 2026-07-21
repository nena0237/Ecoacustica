# ==========================================================
# VALIDACIÓN MATEMÁTICA
#
# Comprueba que TODOS los píxeles de cada rectángulo tengan
# EXACTAMENTE el valor esperado según el CSV.
#
# Si todo está correcto:
#
#        ERROR = 0
#
# ==========================================================

import numpy as np
import torch
from DataLoaderFinal import PlantillaDataset

RUTA_NORMALIZADO = r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\CARACT_NORMALIZADO.csv"
RUTA_IMPORTANCIA = r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\IMPORTANCIA_CARACTERISTICAS.csv"
TAMANO_PIXELES = 128

dataset = PlantillaDataset(ruta_normalizado=RUTA_NORMALIZADO,ruta_importancia=RUTA_IMPORTANCIA,tamano_pixeles=TAMANO_PIXELES)
errores = []
imagenes_a_validar = 20

for indice in range(imagenes_a_validar):
    imagen, etiqueta = dataset[indice]
    imagen = imagen.squeeze(0)
    fila = dataset.df.iloc[indice]
    for r in dataset.rectangulos:
        valor_original = fila.iloc[dataset.N_COLUMNAS_META + r["idx"]]
        gris_esperado = max(0.0, min(1.0, 1.0 - valor_original))
        bloque = imagen[
            r["y1"]:r["y2"],
            r["x1"]:r["x2"]
        ]
        error = torch.abs(
            bloque - gris_esperado
        ).max().item()
        errores.append(error)

errores = np.array(errores)

print(f"\nImágenes evaluadas : {imagenes_a_validar}")
print(f"Rectángulos por imagen : {len(dataset.rectangulos)}")
print(f"Total verificaciones : {len(errores)}")

print("\n================ RESULTADOS ================")
print(f"Error mínimo : {errores.min():.12f}")
print(f"Error medio  : {errores.mean():.12f}")
print(f"Error máximo : {errores.max():.12f}")


if np.allclose(errores, 0.0, atol=1e-12):
    print("\n✅ VALIDACIÓN EXITOSA")
    print("Todos los píxeles contienen EXACTAMENTE el valor esperado.")
    print("Error matemático = 0")
else:
    print("\n❌ VALIDACIÓN FALLIDA")
    print("Se encontraron diferencias.")
    print(f"Mayor diferencia encontrada = {errores.max():.12f}")