import os
from pydub import AudioSegment

# Pasta de entrada (onde estão os arquivos .wav originais)
input_directory = "<diretório da pasta>"

# Pasta de saída (onde os arquivos combinados serão salvos)
output_directory = "<diretório da pasta>"

# Obtém a lista de arquivos .wav na pasta de entrada
wav_files = [file for file in os.listdir(input_directory) if file.endswith(".wav")]

# Dicionário para agrupar arquivos por prefixo de 5 caracteres
file_groups = {}

# Agrupa os arquivos por prefixo de 5 caracteres
for wav_file in wav_files:
    prefix = wav_file[:5]
    if prefix not in file_groups:
        file_groups[prefix] = []
    file_groups[prefix].append(wav_file)

# Concatena e salva os arquivos em cada grupo
for prefix, group in file_groups.items():
    group.sort()  # Ordena alfabeticamente
    combined_audio = None

    for wav_file in group:
        audio = AudioSegment.from_wav(os.path.join(input_directory, wav_file))
        if combined_audio is None:
            combined_audio = audio
        else:
            combined_audio += audio

    # Salva o arquivo combinado com o prefixo na pasta de saída
    output_path = os.path.join(output_directory, f"{prefix}.wav")
    combined_audio.export(output_path, format="wav")

print("Arquivos combinados com sucesso!")
