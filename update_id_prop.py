import pandas as pd

# Set the path to the id_prop.csv file
id_prop_csv_path = "/Users/habibur/Habibur_Python_Scripts/alignn/alignn/id_prop.csv"

# Set the path to the moved_files.csv file
moved_files_csv_path = "/Users/habibur/Habibur_Python_Scripts/alignn/alignn/moved_files.csv"

# Read in the id_prop.csv file as a pandas DataFrame
id_prop_df = pd.read_csv(id_prop_csv_path, index_col=0)

# Read in the moved_files.csv file as a pandas DataFrame
moved_files_df = pd.read_csv(moved_files_csv_path, index_col=0)

# Get a list of the files that were moved
moved_files = moved_files_df.index.tolist()

# Drop the rows in id_prop_df that correspond to the moved files
id_prop_df = id_prop_df.drop(moved_files, errors="ignore")

# Write the updated id_prop.csv file back to disk
id_prop_df.to_csv(id_prop_csv_path)
