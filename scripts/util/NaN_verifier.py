import os
import numpy as np

folder_path = r'C:\caminho\para\sua\pasta'

def check_for_nan(file_path):
    data = np.load(file_path)
    has_nan = np.isnan(data).any()
    return has_nan

files_with_nan = []
for npy_file in os.listdir(folder_path):
    if npy_file.endswith(".npy"):
        file_path = os.path.join(folder_path, npy_file)
        if check_for_nan(file_path):
            files_with_nan.append(npy_file)

print("Arquivos com valores NaN:")
for file_name in files_with_nan:
    print(file_name)

print("Quantidade de arquivos com valores NaN:", len(files_with_nan))
