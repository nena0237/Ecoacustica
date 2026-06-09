import librosa
import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================================
# CONFIGURACIÓN
# ==========================================================

# Carpeta principal de BirdCLEF
CARPETA = r"C:\Users\manue\Downloads\3ccbe\train_audio"

# SOLO PARA PRUEBAS
MAX_FILES = 500

# Cada cuántos audios guardar resultados parciales
GUARDAR_CADA = 50

# Nombre del CSV de salida
ARCHIVO_SALIDA = "features_audios.csv"


# ==========================================================
# FUNCIÓN PARA EXTRAER CARACTERÍSTICAS
# ==========================================================

def extraer_features(ruta_audio):

    # ------------------------------------------------------
    # Carga el audio
    # sr = reduce tiempo de procesamiento
    # mono=True mezcla ambos canales si el audio es estéreo
    # ------------------------------------------------------
    y, sr = librosa.load(
        ruta_audio,
        sr=16000,
        mono=True
    )

    # ------------------------------------------------------
    # RMS
    # Energía promedio del sonido
    # ------------------------------------------------------
    rms = librosa.feature.rms(y=y).mean()

    # ------------------------------------------------------
    # ZCR
    # Número de cruces por cero
    # ------------------------------------------------------
    zcr = librosa.feature.zero_crossing_rate(y).mean()

    # ------------------------------------------------------
    # YIN
    # La calculamos UNA SOLA VEZ
    # ------------------------------------------------------
    yin = librosa.yin(
        y,
        fmin=20,
        fmax=8000
    )

    freq_fundamental = np.mean(yin)
    freq_maxima = np.max(yin)
    freq_minima = np.min(yin)

    # ------------------------------------------------------
    # Características espectrales
    # ------------------------------------------------------
    centroide = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    ).mean()

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    ).mean()

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    ).mean()

    flatness = librosa.feature.spectral_flatness(
        y=y
    ).mean()

    # ------------------------------------------------------
    # MFCC
    # Los 13 coeficientes más usados en audio
    # ------------------------------------------------------
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    mfcc_vals = {
        f"mfcc_{i+1}": mfcc[i].mean()
        for i in range(13)
    }

    # ------------------------------------------------------
    # Construimos una fila del dataset
    # ------------------------------------------------------
    fila = {
        "rms": rms,
        "zcr": zcr,
        "freq_fundamental": freq_fundamental,
        "freq_maxima": freq_maxima,
        "freq_minima": freq_minima,
        "centroide": centroide,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "flatness": flatness,
        **mfcc_vals
    }

    return fila


# ==========================================================
# BUSCAR AUDIOS
# ==========================================================

# Obtiene todas las rutas .ogg
rutas = list(Path(CARPETA).rglob("*.ogg"))

# Solo usamos los primeros para pruebas
rutas = rutas[:MAX_FILES]

print(f"Audios a procesar: {len(rutas)}")


# ==========================================================
# PROCESAMIENTO
# ==========================================================

filas = []

for i, ruta in enumerate(rutas, start=1):

    print(f"[{i}/{len(rutas)}] {ruta.name}")

    try:

        # Extraer características
        features = extraer_features(str(ruta))

        # Nombre archivo
        features["archivo"] = ruta.name

        # La carpeta corresponde a la especie
        features["especie"] = ruta.parent.name

        filas.append(features)

        # --------------------------------------------------
        # Guardado parcial
        # Si algo falla no pierdes todo
        # --------------------------------------------------
        if i % GUARDAR_CADA == 0:

            pd.DataFrame(filas).to_csv(
                ARCHIVO_SALIDA,
                index=False
            )

            print(
                f"Guardado parcial -> "
                f"{len(filas)} audios"
            )

    except Exception as e:

        print(f"ERROR: {e}")


# ==========================================================
# GUARDADO FINAL
# ==========================================================

df = pd.DataFrame(filas)

df.to_csv(
    ARCHIVO_SALIDA,
    index=False
)

print("\nProceso terminado")
print(f"Audios procesados: {len(df)}")

print(df.head())