# CODIGO 5 #

# VAMOS A PINTAR LA PLANTILLA CON ESCALA DE GRISES
# SEGUN LOS VALORES DE LAS CARACTERISTICAS
# aqui se toma la base de datos normalizada y se pintan las caracteristicas
# eso si, se toma la geometria de la plantilla (osea que es como si la volviera a crear)
# entonces no es tan eficiente como se tenia planeado
# este codigo toma la plantilla de relieff


import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from plantilla import geometria, TAMANO_PIXELES, FIGSIZE

RUTA_NORMALIZADO = r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\CARACT_NORMALIZADO.csv"
CARPETA_SALIDA = r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\Imagenes"

ESCALA = 100
N_COLUMNAS_META = 3

os.makedirs(CARPETA_SALIDA, exist_ok=True)
df = pd.read_csv(RUTA_NORMALIZADO)
df = df.head(3)

for indice, fila in df.iterrows():
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=TAMANO_PIXELES)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set_xlim(0,ESCALA)
    ax.set_ylim(0,ESCALA)
    ax.set_aspect("equal")
    ax.axis("off")

    for i in geometria:
        b = geometria[i]
        valor = fila.iloc[N_COLUMNAS_META + i]
        gris = max(0.0, min(1.0, 1.0 - valor))

        ax.add_patch(Rectangle(
                (b["x"]*ESCALA, b["y"]*ESCALA),
                b["w"]*ESCALA,
                b["h"]*ESCALA,
                facecolor=str(gris),
                edgecolor="none",
                linewidth=0
            )
        )

    plt.savefig(os.path.join(CARPETA_SALIDA, f"IMG_{indice}.png"), 
                dpi=TAMANO_PIXELES, bbox_inches=None,pad_inches=0)
    plt.close()

peso_total = 0
peso_total = 0

for archivo in os.listdir(CARPETA_SALIDA):
    if archivo.endswith(".png"):
        peso_total += os.path.getsize(
            os.path.join(CARPETA_SALIDA, archivo)
        )
print(f"Peso total           : {peso_total} bytes ({peso_total/1024:.2f} KB)")