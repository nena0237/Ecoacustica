# ESTE ES EL DATALOADER QUE UTILIZA MATPLOTLIB PARA GENERAR LA IMAGEN
# ESTE CODIGO FUE EL QUE EL PROFE VERIFICO Y SE DIO CUENTA QUE ESTABA BUENO
# PERO ALGO NO ME CUADRABA CON EL CODIGO
# ENTONCES COMO SOY TAN TERCA COMPARE ESTE CODIGO CON OTRO QUE HICE

# ESTE CODIFO LO REENOMBRE FDataLoader.py YA QUE HACE REFERENCIA A FIRST
# A MEDIDA DE ESTE CODIGO IMPLEMENTARE LOS COMENTARIOS DE LAS DIFERENCIAS


# LIBRERIAS #
# ESTE CODIGO IMPLEMENTA MAS LIBRERIAS YA QUE UTILIZA MATPLOTLIB PARA GENERAR LA IMAGEN
# EL CODGIO 2 SOLO TIENE ESTAS: 

import io
import numpy as np # ESTA
import pandas as pd # ESTA
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch # ESTA
from torch.utils.data import Dataset, DataLoader, random_split # ESTA


# CLASE 

class PlantillaDataset(Dataset):

    def __init__(self, ruta_normalizado, ruta_importancia, tamano_pixeles=128, escala=100):
        self.TAMANO_PIXELES = tamano_pixeles
        self.FIGSIZE = (1, 1) # NO ESTA EN EL OTRO CODIGO
        self.ESCALA = escala # NO ESTA EN EL OTRO CODIGO
        self.N_COLUMNAS_META = 3
        self.df = pd.read_csv(ruta_normalizado)
        self.df_imp = pd.read_csv(ruta_importancia)
        ## self._crear_geometria()  ## NUEVO
        ## SELF.RECTANGULOS = [] ## NUEVO

        # ESTA PARTE TIENE OTRA IMPLEMENTACION DIFERENTE
        # EN VEZ DE ESTO:
        self.ancho_px = int(self.FIGSIZE[0] * self.TAMANO_PIXELES)
        self.alto_px = int(self.FIGSIZE[1] * self.TAMANO_PIXELES)

        # ES ESTO:
        # for idx, b in self.geometria.items():
        #     x1 = int(b["x"] * self.TAMANO_PIXELES)
        #     y1 = int(b["y"] * self.TAMANO_PIXELES)
        #     x2 = int((b["x"] + b["w"]) * self.TAMANO_PIXELES)
        #     y2 = int((b["y"] + b["h"]) * self.TAMANO_PIXELES)
        # self.rectangulos.append({'idx': idx,'x1': x1, 'y1': y1,'x2': x2, 'y2': y2})


## AQUI NO EXISTE FUNCION _CREAR_GEOMETRIA
# PERO IGUAL ESTAS LINEAS DE CODIGO SON IGUALES A LAS DEL OTRO CODIGO

        areas = self.df_imp.iloc[:,2].tolist()

        if len(areas) % 3 != 0:
            raise ValueError("Numero de caracteristicas debe ser multiplo de 3.")

        self.geometria = {}
        y = 0.0

        for k in range(0,len(areas),3):
            fila = [k,k+1,k+2]
            altura = sum(areas[i] for i in fila)
            x = 0.0

            for i in fila:
                ancho = areas[i]/altura
                self.geometria[i] = {
                    "x":x,
                    "y":y,
                    "w":ancho,
                    "h":altura
                }
                x += ancho
            y += altura

# HASTA AQUI VAN EXACTAMENTE IGUALES
# MISMA ESTRUCTURA DE GEOMETRIA

        print("="*60)
        print(f"Tamano imagen : {self.ancho_px}x{self.alto_px}")
        print(f"Total registros : {len(self.df)}")
        print("Plantilla creada.")
        print("="*60)


# A PARTIR DE AQUI EL OTRO CODIGO IMPLEMENTA LA FUNCION _GENERAR_IMAGEN_PIXEL 
# Es justamente es la razón por la que tu SDataLoader obtuvo error matemático = 0.
# Aqui no era necesario porque Matplot ya hace ese trabajo internamente (rasterizacion)
# En el otro tocaba hacerlo a mano ya que no estamos dibujando 


# MISMO LEN

    def __len__(self):
        return len(self.df)


# EL GETITEM LINEA 1 Y 2 IGUALES

    def __getitem__(self,index):
        fila = self.df.iloc[index]

        # AQUI YA ES DIFERENTE, EN EL OTRO CODIGO SON:
        # imagen = self._generar_imagen_pixel(fila)
        # etiqueta = fila.iloc[1]
        # return imagen, etiqueta

        fig,ax = plt.subplots(figsize=self.FIGSIZE, dpi=self.TAMANO_PIXELES)
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

        ax.set_xlim(0,self.ESCALA)
        ax.set_ylim(0,self.ESCALA)
        ax.set_aspect("equal")
        ax.axis("off")

        for i,b in self.geometria.items():
            valor = fila.iloc[self.N_COLUMNAS_META+i]
            gris = max(0.0,min(1.0,1.0-valor))

            ax.add_patch(
                Rectangle(
                    (b["x"]*self.ESCALA,b["y"]*self.ESCALA),
                    b["w"]*self.ESCALA,
                    b["h"]*self.ESCALA,
                    facecolor=str(gris),
                    edgecolor="none"
                )
            )

        fig.canvas.draw()
        imagen = np.asarray(fig.canvas.buffer_rgba())[:,:,:3]
        plt.close(fig)
        imagen = torch.from_numpy(imagen).permute(2,0,1).float()/255.
        etiqueta = fila.iloc[1]
        return imagen, etiqueta

## AQUI ES IGUAL
# EN EL OTRO CODIGO LO BORRE PORQUE NO ERA NECESARIO
if __name__ == "__main__":

    RUTA_NORMALIZADO = r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\CARACT_NORMALIZADO.csv"
    RUTA_IMPORTANCIA = r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\IMPORTANCIA_CARACTERISTICAS.csv"
    BATCH_SIZE = 20
    # TAMANO_PIXELES = 128

    dataset = PlantillaDataset(RUTA_NORMALIZADO, RUTA_IMPORTANCIA)
    #TAMANO_PIXELES

    n_train = int(0.7*len(dataset))
    n_val = len(dataset)-n_train
    train_dataset, val_dataset = random_split(dataset,[n_train,n_val])
    train_loader = DataLoader( train_dataset, batch_size=BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Entrenamiento : {len(train_dataset)}")
    print(f"Validacion    : {len(val_dataset)}")

    imagenes, etiquetas = next(iter(train_loader))
    print("Primer batch")
    print("Imagenes :", imagenes.shape)
    print("Etiquetas:", etiquetas.shape)