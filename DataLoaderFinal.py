### ESTE ES EL DATALOADER FINAL QUE SE UTILIZARA PARA ENTRENAR LOS MODELOS
# LOS CAMBIOS QUE SE LE INCORPORARON FUERON LOS SIGUIENTES:

# Cambio 1. N_COLUMNAS_META
# Original:
# self.N_COLUMNAS_META = 3
# Nuevo:
# self.N_COLUMNAS_META = 4

# Cambio 2. Agregar información de las clases
# Original:
# No existía.
# Nuevo:
# self.classes = self.df["scientific_name"].unique()
# self.n_classes = len(self.classes)

# Cambio 3. Imprimir número de clases
# Original:
# print(f"Total registros : {len(self.df)}")
# print(f"Total rectángulos : {len(self.rectangulos)}")
# Nuevo:
# print(f"Total registros : {len(self.df)}")
# print(f"Número de clases : {self.n_classes}")
# print(f"Total rectángulos : {len(self.rectangulos)}")

# Cambio 4. Devolver el label numérico
# Original:
# etiqueta = fila.iloc[1]
# Nuevo:
# etiqueta = torch.tensor(fila.iloc[2],dtype=torch.long)

# Cambio 5. Convertir la etiqueta en tensor
# Original:
# etiqueta = fila.iloc[1]
# Nuevo:
# torch.tensor(..., dtype=torch.long)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class PlantillaDataset(Dataset):

    def __init__(self, ruta_normalizado, ruta_importancia, tamano_pixeles=128):
        self.TAMANO_PIXELES = tamano_pixeles

        # Son 4 columnas porque ahora existen:
        # 1. scientific_name
        # 2. class
        # 3. label
        # 4. características

        self.N_COLUMNAS_META = 4

        self.df = pd.read_csv(ruta_normalizado)
        self.df_imp = pd.read_csv(ruta_importancia)

        # Parte nueva 
        self.classes = self.df["scientific_name"].unique()
        self.n_classes = len(self.classes)
        # =========================

        self._crear_geometria()
        self.rectangulos = []

        for idx, b in self.geometria.items():
            x1 = int(b["x"] * self.TAMANO_PIXELES)
            y1 = int(b["y"] * self.TAMANO_PIXELES)
            x2 = int((b["x"] + b["w"]) * self.TAMANO_PIXELES)
            y2 = int((b["y"] + b["h"]) * self.TAMANO_PIXELES)
            self.rectangulos.append({
                "idx": idx,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

        print("=" * 60)
        print(f"Tamaño imagen : {self.TAMANO_PIXELES}x{self.TAMANO_PIXELES}")
        print(f"Total registros : {len(self.df)}")
        print(f"Número de clases : {self.n_classes}")
        print(f"Total rectángulos : {len(self.rectangulos)}")
        print("Plantilla creada (generación pixel a pixel).")
        print("=" * 60)

    def _crear_geometria(self):
        areas = self.df_imp.iloc[:, 2].tolist()

        if len(areas) % 3 != 0:
            raise ValueError("Número de características debe ser múltiplo de 3.")

        self.geometria = {}
        y = 0.0

        for k in range(0, len(areas), 3):
            fila = [k, k + 1, k + 2]
            altura = sum(areas[i] for i in fila)
            x = 0.0

            for i in fila:
                ancho = areas[i] / altura
                self.geometria[i] = {
                    "x": x,
                    "y": y,
                    "w": ancho,
                    "h": altura
                }
                x += ancho
            y += altura

    def _generar_imagen_pixel(self, fila):
        imagen = np.ones((self.TAMANO_PIXELES, self.TAMANO_PIXELES), dtype=np.float32)

        for r in self.rectangulos:
            valor = fila.iloc[self.N_COLUMNAS_META + r["idx"]]
            gris = max(0.0, min(1.0, 1.0 - valor))
            imagen[r["y1"]:r["y2"], r["x1"]:r["x2"]] = gris

        imagen_tensor = (torch.from_numpy(imagen).unsqueeze(0).float())
        return imagen_tensor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        fila = self.df.iloc[index]
        imagen = self._generar_imagen_pixel(fila)

        # Parte nueva 
        # columna label
        etiqueta = torch.tensor(fila.iloc[2], dtype=torch.long)
        # =========================

        return imagen, etiqueta