import os
def rename_files(path):
    for filename in os.listdir(path):
        if filename.endswith('.png'):
            parts = filename.split('_')
            last_part = parts[-1].split('.')[0]
            new_last_part = last_part.zfill(3)
            new_filename = '_'.join(parts[:-1]) + '_' + new_last_part + '.png'
            os.rename(os.path.join(path, filename), os.path.join(path, new_filename))

path = 'D:\\Raindrop_folder\\Rainfall_project_2023\\rainData\\RR-20231206-outdoor\\algorithm1_out_R'
rename_files(path)
