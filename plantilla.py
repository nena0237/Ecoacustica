# CODIGO 4 #

# PLANTILLA GEOMETRICA PARA TODAS LAS CARACTERISTICAS
# aqui se crea un diccionario con la geometria de cada caracteristica
# aqui se debe cambiar el tamaño de pixeles 
# este codigo se esta ejecutando con relieff

import pandas as pd 

TAMANO_PIXELES = 128
FIGSIZE = (1,1)

df_imp = pd.read_csv("C:\\Users\\manue\\Downloads\\3ccbe\\Proyecto_final\\IMPORTANCIA_CARACTERISTICAS_RELIEFF.csv")
df_normalizado = pd.read_csv(r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\CARACT_NORMALIZADO.csv")

ancho_px = int(FIGSIZE[0] * TAMANO_PIXELES)
alto_px = int(FIGSIZE[1] * TAMANO_PIXELES)

areas = df_imp.iloc[:,2].tolist()
etiquetas = df_imp.iloc[:,1].tolist()

n = len(areas)

if n % 3 != 0:
    raise ValueError("Este codigo espera un numero de caracteristicas multiplo de 3.")

filas = []
for i in range(0,n,3):
    filas.append(list(range(i,i+3)))

geometria = {}

y = 0.0

for fila in filas:

    altura = sum(areas[i] for i in fila)

    x = 0.0

    for i in fila:

        ancho = areas[i]/altura

        geometria[i]={
            "x":x,
            "y":y,
            "w":ancho,
            "h":altura
        }

        x += ancho

    y += altura

print(f"Tamaño de imagen     : {ancho_px}x{alto_px} px")
print("Plantilla geométrica creada.")