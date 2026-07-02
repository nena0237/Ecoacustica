# CODIGO 1 #

# Es el primero que debemos utilizar para poder empezar a programar.
# este codigo se encarga de filtrar los datos que se encuentran en el archivo csv
# TRAIN Y TAXONOMY ORIGINALES 
# para poder obtener solo los datos que necesitamos para nuestro proyecto
# Que son aves y anfibios y los audios menores a 10 segundos

# este es el unico codigo que no esta automatizado :)


import os
import pandas as pd
import soundfile as sf

RUTA_TAXONOMY = r"C:\Users\manue\Downloads\3ccbe\DATOS\taxonomy.csv"
RUTA_TRAIN = r"C:\Users\manue\Downloads\3ccbe\DATOS\train.csv"

CARPETA_AUDIOS = r"C:\Users\manue\Downloads\3ccbe\train_audio"

SALIDA_TAXONOMY = r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\filtradoTAXO.csv"
SALIDA_TRAIN = r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\filtradoTRAIN.csv"
SALIDA_10S = r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\filtrado10s.csv"

print("\n" + "=" * 60)
print("FILTRANDO TAXONOMY")
print("=" * 60)

# parte 1 #

taxonomy = pd.read_csv(RUTA_TAXONOMY)

total_original_taxo = len(taxonomy)

taxonomy_filtrado = taxonomy[taxonomy["class_name"].isin(["Aves", "Amphibia"])].copy()

total_final_taxo = len(taxonomy_filtrado)

eliminados_taxo = total_original_taxo - total_final_taxo

aves_taxo = (taxonomy_filtrado["class_name"] == "Aves").sum()
amphibia_taxo = (taxonomy_filtrado["class_name"] == "Amphibia").sum()

taxonomy_filtrado.to_csv(SALIDA_TAXONOMY,index=False)

print(f"Registros originales : {total_original_taxo}")
print(f"Registros eliminados : {eliminados_taxo}")
print(f"Registros finales    : {total_final_taxo}")
print(f"Aves                 : {aves_taxo}")
print(f"Amphibia             : {amphibia_taxo}")
print(f"Archivo generado     : {SALIDA_TAXONOMY}")

# parte 2 #

print("\n" + "=" * 60)
print("ETAPA 2 - FILTRANDO TRAIN")
print("=" * 60)

train = pd.read_csv(RUTA_TRAIN)

total_original_train = len(train)

columnas_union = ["primary_label","scientific_name","common_name"]

claves_validas = taxonomy_filtrado[["primary_label", "scientific_name", "common_name", "class_name"]].drop_duplicates()

train_filtrado = train.merge(claves_validas,
    on=["primary_label", "scientific_name", "common_name"],how="inner")

total_final_train = len(train_filtrado)

eliminados_train = total_original_train - total_final_train

conteo_clases_train = (train_filtrado["scientific_name"].map
    (taxonomy_filtrado.set_index
    ("scientific_name")["class_name"]).value_counts().to_dict())

aves_train = conteo_clases_train.get("Aves", 0)

amphibia_train = conteo_clases_train.get("Amphibia", 0)

train_filtrado.to_csv(SALIDA_TRAIN,index=False)

print(f"Registros originales : {total_original_train}")
print(f"Registros eliminados : {eliminados_train}")
print(f"Registros finales    : {total_final_train}")
print(f"Audios Aves          : {aves_train}")
print(f"Audios Amphibia      : {amphibia_train}")
print(f"Archivo generado     : {SALIDA_TRAIN}")

# parte 3 #

print("\n" + "=" * 60)
print("ETAPA 3 - ANALIZANDO DURACION DE AUDIOS")
print("=" * 60)

registros_validos = []

audios_revisados = 0
audios_eliminados_duracion = 0
audios_no_encontrados = 0
audios_error_lectura = 0

for fila in train_filtrado.itertuples(index=False):

    ruta_audio = os.path.join(CARPETA_AUDIOS, fila.filename)
    audios_revisados += 1

    if not os.path.exists(ruta_audio):
        audios_no_encontrados += 1
        continue

    try:
        duracion = sf.info(ruta_audio).duration
    except Exception:
        audios_error_lectura += 1
        continue

    if duracion > 10:
        audios_eliminados_duracion += 1
        continue

    registros_validos.append({**fila._asdict(),
    "ruta_audio": ruta_audio,"duracion_seg": round(duracion, 4)})

    if audios_revisados % 1000 == 0:
        print(f"Audios revisados: {audios_revisados}")

# parte 4 #

df_final = pd.DataFrame(registros_validos)


columnas_prioridad = ["ruta_audio","scientific_name","class_name","duracion_seg"]

columnas_existentes = [c for c in columnas_prioridad if c in df_final.columns]

otras_columnas = [c for c in df_final.columns if c not in columnas_existentes]

df_final = df_final[columnas_existentes + otras_columnas]

df_final.to_csv(SALIDA_10S,index=False)

conteo_final = (df_final["class_name"].value_counts().to_dict())

aves_final = conteo_final.get("Aves", 0)

amphibia_final = conteo_final.get("Amphibia", 0)

print("\n" + "=" * 60)
print("REPORTE FINAL")
print("=" * 60)

print(f"Audios revisados                : {audios_revisados}")
print(f"Audios eliminados (>10 seg)     : {audios_eliminados_duracion}")
print(f"Audios no encontrados           : {audios_no_encontrados}")
print(f"Audios con error de lectura     : {audios_error_lectura}")
print(f"Audios finales                  : {len(df_final)}")
print(f"Audios Aves                     : {aves_final}")
print(f"Audios Amphibia                 : {amphibia_final}")

print("\nArchivos generados:")
print(SALIDA_TAXONOMY)
print(SALIDA_TRAIN)
print(SALIDA_10S)

print("\nPROCESO TERMINADO")