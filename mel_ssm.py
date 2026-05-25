# MEL SPECTROGRAM + SSM

import numpy as np 
import matplotlib.pyplot as plt 
import librosa 
import librosa.display 
import os 
import pandas as pd

# CONFIGURACIÓN 
DATA_PATH = r"C:\Users\manue\Downloads\3ccbe"

# Cargar el CSV y tomar un audio de ejemplo 
df_train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))

# Seleccionamos un audio
primera_fila = df_train.iloc[56]
ruta_audio = os.path.join(DATA_PATH, "train_audio",
primera_fila["filename"])

print(f"Especie: {primera_fila['primary_label']}") 
print(f"Archivo: {ruta_audio}")

# ---- CARGAR EL AUDIO ----
señal, sr = librosa.load(ruta_audio, sr=32000, duration=10)
print(f"Duración cargada: {len(señal)/sr:.1f} segundos")
print(f"Frecuencia de muestreo: {sr} Hz")

# ---- MEL SPECTROGRAM ----

mel = librosa.feature.melspectrogram(
    y=señal, sr=sr,n_mels=128, n_fft=2048, hop_length=512, fmax=16000)
mel_db = librosa.power_to_db(mel, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_db, sr=sr, 
    hop_length=512, x_axis='time', y_axis='mel', fmax=16000, cmap='magma')

plt.colorbar(format='%+2.0f dB')
plt.title(f'MEL SPECTROGRAM — {primera_fila["primary_label"]}')
plt.tight_layout()
plt.savefig('mel_spectrogram.png', dpi=150)
plt.show()
print("Guardado: mel_spectrogram.png")

# REPRESENTACIÓN 2: SELF-SIMILARITY MATRIX (SSM)

mfcc = librosa.feature.mfcc(y=señal, sr=sr, n_mfcc=20, hop_length=512)

from sklearn.preprocessing import normalize 

mfcc_norm = normalize(mfcc.T)
SSM = np.dot(mfcc_norm, mfcc_norm.T)
plt.figure(figsize=(7, 6))
plt.imshow(SSM, origin='lower', aspect='auto', cmap='inferno', vmin=0, vmax=1)
plt.colorbar(label='Similitud')
plt.title(f'SELF-SIMILARITY MATRIX — {primera_fila["primary_label"]}')
plt.xlabel('Tiempo (frames)')
plt.ylabel('Tiempo (frames)')
plt.tight_layout()
plt.savefig('ssm_patron_franja.png', dpi=150)
plt.show()
print("Guardado: ssm_patron_franja.png")

# COMPARACIÓN VISUAL LADO A LADO

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(mel_db, aspect='auto', origin='lower', cmap='magma')
axes[0].set_title('Mel Spectrogram', fontsize=13)
axes[0].set_xlabel('Tiempo (frames)')
axes[0].set_ylabel('Frecuencia (mel)') 

axes[1].imshow(SSM, aspect='auto', origin='lower', 
    cmap='inferno', vmin=0, vmax=1)
axes[1].set_title('Self-Similarity Matrix\n(stripes)', fontsize=13)
axes[1].set_xlabel('Tiempo')
axes[1].set_ylabel('Tiempo')

plt.suptitle(f'Especie: {primera_fila["primary_label"]}', 
    fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('comparacion_representaciones.png', dpi=150)
plt.show()
print("Guardado: comparacion_representaciones.png")