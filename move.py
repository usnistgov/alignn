import os
import random
import shutil
import pandas as pd

# set the path to the folder containing the .cif files and csv file
folder_path = "/Users/habibur/Habibur_Python_Scripts/alignn/alignn/DataSet_B"

# set the path to the new folder where you want to move the selected files
new_folder_path = "/Users/habibur/Habibur_Python_Scripts/alignn/alignn/20%/"

# set the percentage of files you want to select
percent_to_select = 10

# read the csv file into a pandas dataframe
csv_file_path = os.path.join(folder_path, "id_prop.csv")
df = pd.read_csv(csv_file_path, index_col=0)

# get a list of all the files in the folder with the .cif extension
file_list = [f for f in os.listdir(folder_path) if f.endswith(".cif")]

# calculate the number of files to select
num_to_select = int(len(file_list) * (percent_to_select / 100))

# randomly select the files to move
files_to_move = random.sample(file_list, num_to_select)

# move the selected files to the new folder
moved_files = []
for file_name in files_to_move:
    file_path = os.path.join(folder_path, file_name)
    new_file_path = os.path.join(new_folder_path, file_name)
    shutil.move(file_path, new_file_path)
    moved_files.append(file_name)

# create a new dataframe with the moved files and their corresponding values
moved_df = df.loc[moved_files]

# write the moved files and values to a new csv file
moved_csv_path = os.path.join(new_folder_path, "moved_files.csv")
moved_df.to_csv(moved_csv_path)










