# CODIGO 2 #

# EXTRACCION DE CARACTERISTICAS + NORMALIZACION
# Las caracterisitcas se sacan directamente desde librosa, entonces es 100% confiable
# las unicas que no fueron con librosa son las frecuencias
# para sacar las frecuencias se utlizo el Raven Pro (es un estudio docuemntado)
# que se sacaron de la energia acumulada de la STFT

# POR ESO: 
# freq_minima:
#   frecuencia donde la energia acumulada cruza 5%
# freq_maxima:
#   frecuencia donde la energia acumulada cruza 95%
# Se calcula UNA sola STFT por audio
# y se reutiliza para todas las caracteristicas.


import os
import gc
import librosa
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

RUTA_CSV = r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\filtrado10s.csv"

SALIDA_BRUTO = r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\CARACT_BRUTO.csv"

SALIDA_NORMALIZADO = r"C:\Users\manue\Downloads\3ccbe\Proyecto_final\CARACT_NORMALIZADO.csv"

# Parte 1 - EXTRACCION CARACTERISTICAS #

print("=" * 60)
print("CARGANDO LISTA DE AUDIOS")
print("=" * 60)

df = pd.read_csv(RUTA_CSV)

print(f"Audios encontrados : {len(df)}")

resultados = []

procesados = 0
errores = 0

for fila in df.itertuples(index=False):

    ruta_audio = fila.ruta_audio
    scientific_name = fila.scientific_name
    class_name = fila.class_name

    try:
        y, sr = librosa.load(ruta_audio, sr=22050, mono=True)
        
        # STFT UNICA
        S = np.abs(librosa.stft(y,n_fft=2048,hop_length=512))
        frecuencias = librosa.fft_frequencies(sr=sr,n_fft=2048)
        
        # ENERGIA POR FRECUENCIA
        energia_por_frecuencia = np.sum( S ** 2,axis=1)
        energia_total = energia_por_frecuencia.sum()
        energia_acumulada = np.cumsum(energia_por_frecuencia) / energia_total

        # FREQ MINIMA (5%)
        indice_min = np.searchsorted(energia_acumulada, 0.05)
        freq_minima = float(frecuencias[indice_min])

        # FREQ MAXIMA (95%)
        indice_max = np.searchsorted(energia_acumulada,0.95)
        freq_maxima = float(frecuencias[indice_max])
        
        # CENTROIDE
        centroide = float( np.mean(librosa.feature.spectral_centroid(S=S,sr=sr)))

        # BANDWIDTH
        bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=S,sr=sr)))

        # ROLLOFF
        rolloff = float( np.mean(librosa.feature.spectral_rolloff(S=S,sr=sr)))

        # SPECTRAL CONTRAST
        spectral_contrast = float( np.mean(librosa.feature.spectral_contrast(S=S,sr=sr)))

        # FLATNESS
        flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))

        # RMS
        rms = float(np.mean(librosa.feature.rms(S=S)))

        # ZCR
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        resultados.append({
            "ruta_audio": ruta_audio,
            "scientific_name": scientific_name,
            "class_name": class_name,
            "freq_minima": freq_minima,
            "freq_maxima": freq_maxima,
            "centroide": centroide,
            "bandwidth": bandwidth,
            "rolloff": rolloff,
            "spectral_contrast": spectral_contrast,
            "flatness": flatness,
            "rms": rms,
            "zcr": zcr
        })

        procesados += 1

        if procesados % 500 == 0:
            print(f"Audios procesados: {procesados}")

        del y
        del S
        del energia_por_frecuencia
        del energia_acumulada
        gc.collect()

    except Exception as e:
        errores += 1
        print(f"\nERROR: {ruta_audio}")
        print(e)

df_bruto = pd.DataFrame(resultados)
df_bruto.to_csv(SALIDA_BRUTO,index=False)

print("\n" + "=" * 60)
print("CARACTERISTICAS EXTRAIDAS")
print("=" * 60)

print(f"Audios procesados : {procesados}")
print(f"Errores           : {errores}")
print(f"Archivo generado  : {SALIDA_BRUTO}")


# Parte 2 - NORMALIZACION

print("\n" + "=" * 60)
print("NORMALIZANDO")
print("=" * 60)

df_norm = df_bruto.copy()
columnas_caracteristicas = [
    "freq_minima",
    "freq_maxima",
    "centroide",
    "bandwidth",
    "rolloff",
    "spectral_contrast",
    "flatness",
    "rms",
    "zcr"
]

scaler = MinMaxScaler()
df_norm[columnas_caracteristicas] = scaler.fit_transform(df_norm[columnas_caracteristicas])
df_norm.to_csv(SALIDA_NORMALIZADO,index=False)

print("\n" + "=" * 60)
print("REPORTE FINAL")
print("=" * 60)

print(f"CSV bruto        : {len(df_bruto)} registros")
print(f"CSV normalizado  : {len(df_norm)} registros")

print("\nArchivos creados:")
print(SALIDA_BRUTO)
print(SALIDA_NORMALIZADO)

print("\nPROCESO TERMINADO")