import os
import librosa
import numpy as np
import pandas as pd
import librosa.display

# Seu código de extração de características
def feature_extraction(file_path):
    # Carregar o arquivo de áudio
    x, sample_rate = librosa.load(file_path, res_type="kaiser_fast")
    # Extair características dos áudios
    mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=50).T, axis=0)
    
    return mfcc

features = {}
directory = r"C:\Users\Lenovo\Desktop\IC\[99] Database Final (16000_0)\trimmed_audio\cookie_d\combined_audio"

# Criar um DataFrame vazio
df = pd.DataFrame()

for audio in os.listdir(directory):
    audio_path = directory + "\\" + audio
    # Calcular características
    audio_features = feature_extraction(audio_path)
    # Use o nome do arquivo como índice
    audio_name = audio.split(".")[0]  # Remove a extensão .wav
    df[audio_name] = audio_features

# Adicionar um cabeçalho para a primeira coluna (nomes dos arquivos)
df = df.rename(columns={df.columns[0]: "nome do arquivo"})

# Transpor o DataFrame
df = df.transpose()

# Salvar o DataFrame como um arquivo CSV
df.to_csv("features2.csv", index=False)
