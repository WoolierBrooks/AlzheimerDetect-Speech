import os
import numpy as np

# Pasta com os arquivos dos pacientes controle
folder_control = "<diretório da pasta>"

# Pasta com os arquivos dos pacientes positivos
folder_positive = "<diretório da pasta>"

# Pasta para salvar os arquivos preenchidos
output_folder_control = "<diretório da pasta>"
output_folder_positive = "<diretório da pasta>"

# Número máximo de pontos desejado (4296)
max_points = 4296

# Função para preencher um array até o número máximo de pontos
def fill_array(array, target_length):
    if len(array) < target_length:
        average = np.mean(array)
        padding_length = target_length - len(array)
        padding = [average] * padding_length
        array = np.concatenate((array, padding))
    return array

# Loop para processar os arquivos dos pacientes controle
for npy_file in os.listdir(folder_control):
    if npy_file.endswith(".npy"):
        input_file_path = os.path.join(folder_control, npy_file)
        output_file_path = os.path.join(output_folder_control, npy_file)
        data = np.load(input_file_path)
        filled_data = fill_array(data, max_points)
        np.save(output_file_path, filled_data)

# Loop para processar os arquivos dos pacientes positivos
for npy_file in os.listdir(folder_positive):
    if npy_file.endswith(".npy"):
        input_file_path = os.path.join(folder_positive, npy_file)
        output_file_path = os.path.join(output_folder_positive, npy_file)
        data = np.load(input_file_path)
        filled_data = fill_array(data, max_points)
        np.save(output_file_path, filled_data)
