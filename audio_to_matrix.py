import librosa
import numpy as np
import os

# Carregar e converter audio em vetor
def carregar_audio(rota_arquivo):
    audio, _ = librosa.load(rota_arquivo, sr=None)
    return audio.astype(np.float32)

# Diretórios das pastas "Control" e "Dementia"
pasta_control = r'C:\caminho\para\sua\pasta\Control'
pasta_dementia = r'C:\caminho\para\sua\pasta\Dementia'

# Lista de pastas dentro de Control e Dementia
pastas = ['cookie', 'fluency', 'recall', 'sentence']

# Percorrer as pastas e carregar os dados de audio de cada paciente
for pasta in pastas:
    # Diretório da pasta atual (Control)
    dir_control = os.path.join(pasta_control, pasta)
    
    # Diretório da pasta atual (Dementia)
    dir_dementia = os.path.join(pasta_dementia, pasta)
    
    # Obter a lista de arquivos na pasta atual (Control)
    arquivos_control = os.listdir(dir_control)
    
    # Obter a lista de arquivos na pasta atual (Dementia)
    arquivos_dementia = os.listdir(dir_dementia)

    # Percorrer os arquivos em Control para carregar os dados de audio
    for arquivo_control in arquivos_control:
        if arquivo_control.endswith(".wav"):
            rota_control = os.path.join(dir_control, arquivo_control)
        
            # Carregar o audio de Control em vetor
            audio_control = carregar_audio(rota_control)     

            # Obter o caminho completo para o arquivo de saída .txt
            output_dir = os.path.join(dir_control, "txt_files")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, os.path.splitext(arquivo_control)[0] + ".txt")

            # Salvar apenas o array no arquivo .txt
            with open(output_file, "w") as f:
                np.set_printoptions(threshold=np.inf)  # Definir opções de exibição para imprimir o array completo
                f.write(np.array2string(audio_control, separator=', '))

    # Percorrer os arquivos em Dementia para carregar os dados de audio
    for arquivo_dementia in arquivos_dementia:
        if arquivo_dementia.endswith(".wav"):
            rota_dementia = os.path.join(dir_dementia, arquivo_dementia)

            # Carregar o audio de Dementia em vetor
            audio_dementia = carregar_audio(rota_dementia)

            # Obter o caminho completo para o arquivo de saída .txt
            output_dir = os.path.join(dir_dementia, "txt_files")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, os.path.splitext(arquivo_dementia)[0] + ".txt")

            # Salvar apenas o array no arquivo .txt
            with open(output_file, "w") as f:
                np.set_printoptions(threshold=np.inf)  # Definir opções de exibição para imprimir o array completo
                f.write(np.array2string(audio_dementia, separator=', '))