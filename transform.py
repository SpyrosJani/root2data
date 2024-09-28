import uproot
import h5py
import os
import re
import numpy as np
import pandas as pd
import awkward as ak #pip install akward-pandas 
from typing import List, Dict



def get_tree_branches(root_file, columns_to_find) -> Dict[str, List[str]]: 
    """
    Extracts and returns the columns of interest from a ROOT file. It identifies the tree containing the branches(columns_to_find)
    and returns a dictionary mapping tree names to the list of these columns.

    Parameters:
     - root_file (str): The file path to the ROOT file.
     - cols_to_find (List[str]): A list of columns to extract from the ROOT file.

    Returns:
     - tree_dict: A dictionary where the keys are tree names and the values are lists of columns found in those trees.
    """
    tree_dict = {}
    for tree_name in root_file.keys():
        tree = root_file[tree_name]
        if all(col in tree.keys() for col in columns_to_find): # check if all requested columns are in the tree
            tree_list = [col for col in columns_to_find if col in tree.keys()] # get a list of the columns that are in the tree: phaseIITriggerTree
            tree_dict[tree_name] = tree_list
            print(f"Found columns: {tree_list} in tree: {tree_name}")
        else:
            print(f"Columns not found in tree: {tree_name}")
    return tree_dict


def root_to_dict_of_arrays(root_files_path: str, columns_to_find: List[str]) -> Dict[str, np.ndarray]:
    """
    Extract data from a ROOT file and return it as a dictionary of NumPy arrays.

    Parameters:
     - root_files_path : str
     - columns_to_find : List[str]

     Returns:
     - Dict[str, np.ndarray]
    """
    root_file = uproot.open(root_files_path)
    tree_and_branches = get_tree_branches(root_file, columns_to_find)
    dict_of_arrays = {}
    list_of_arrays = [] # TODO: remove this
    for tree_name, cols in tree_and_branches.items():
        # root_file[tree_name].arrays(cols, library='np') # exp1
        data = root_file[tree_name].arrays(cols) # exp2 returns <class 'awkward.highlevel.Array'>
        list_of_arrays.append(data) # TODO remove this
        dict_of_arrays.update(root_file[tree_name].arrays(cols, library="np")) # correct!
        # return dict_of_arrays, list_of_arrays # TODO remove this
    return dict_of_arrays

    for tree_name, cols in tree_and_branches.items():
        arrays = root_file[tree_name].arrays(cols, library="np")
    # arrays = root_file[tree_name].arrays(columns_to_find, library="np") # tree name is fetched from the dictionary.keys()
    return arrays


def root_to_awkward_arrays(root_files_path: str, columns_to_find: List[str]) :
    """
    Extract data from a ROOT file and return it as a dictionary of NumPy arrays.

    Parameters:
     - root_files_path : str
     - columns_to_find : List[str]

     Returns:
     - Dict[str, np.ndarray]
    """
    root_file = uproot.open(root_files_path)
    tree_and_branches = get_tree_branches(root_file, columns_to_find)
    for tree_name, cols in tree_and_branches.items():
        # root_file[tree_name].arrays(cols, library='np') # exp1
        awkward_data = root_file[tree_name].arrays(cols) # exp2 returns <class 'awkward.highlevel.Array'>
        # return dict_of_arrays, list_of_arrays # TODO remove this
    return awkward_data


def save_to_sqlite(arrays: Dict[str, np.ndarray], awkward_arrays, h5_file_path: str) -> None:
    pass

def save_to_h5(arrays: Dict[str, np.ndarray], awkward_arrays, h5_file_path: str) -> None:

##########################################################################################
#      SCENARIO 3 - convert byte string data to list of arrays and store as variable-length sequence       
##########################################################################################
    with h5py.File(h5_file_path, 'w') as h5_file:
        for key, array in arrays.items():
            if array.dtype == object:
                try:
                    array = np.array([np.array(item, dtype=np.float64) for item in array])
                except ValueError:
                    list_of_arrays = []
                    print('Can not convert to float64. Converting to byte strings')
                    array_with_byte_data = np.array([str(item) for item in array], dtype='S')
                    for item in array_with_byte_data:
                        if len(str(item)) > 1:
                            string_data = item.decode('utf-8').replace('[', '').replace(']', '').replace('\n', '').strip()
                            string_data = re.sub(r'\s+', ' ', string_data).split(' ')
                            float_data = [float(x) for x in string_data]
                            list_of_arrays.append(np.array(float_data, dtype=np.float64)) 
                    ##TODO: store the list of arrays as a variable-length sequence <!>
                    dt = h5py.special_dtype(vlen=np.dtype('float64'))
                    array = np.array(list_of_arrays, dtype=dt)
            else:
                array = array.astype(np.float64)
            h5_file.create_dataset(key, data=array)
    print(f'data has been successfully written to {h5_file_path}')
    return None


def read_h5_file(h5_file_path: str) -> None:
    with h5py.File(h5_file_path, 'r') as h5_file:
        print(f"Contents of {h5_file_path}:")
        for column in h5_file.keys():
            if isinstance(h5_file[column], h5py.Group):
                print(f"Group: {column}")
                for subkey in h5_file[column].keys():
                    data = h5_file[column][subkey][:]
                    print(f"\nDataset_subkey: {subkey}")
                    print(data)
            else:
                data = h5_file[column][:]
                print(f"\nDataset: {column}")
                print(data)

# OLD but worth exploring
def root_to_h5(root_files_path: str, cols_to_find: List[str], h5_file: str) -> np.ndarray:   
    """
    Convert a ROOT file to an HDF5 file.

    * Parameters:
        root_file_path : str
        h5_file_path : str

    * Returns:
        None
    """
    # Open the ROOT file
    root_file = uproot.open(root_files_path)

    for tree_name in root_file.keys():
        print(f"Processing tree: {tree_name}")
        tree = root_file[tree_name]
        
        # Get all branches
        branches = tree.keys()
        
        # Create a group for each tree
        tree_group = h5_file.create_group(tree_name)
        
        # Read all data from the tree
        data = tree.arrays()
        
        # Iterate through branches
        for branch in branches:
            branch_data = data[branch]
            
            # Convert to numpy array if possible
            try:
                np_data = ak.to_numpy(branch_data)
                # If successful, store as a dataset
                tree_group.create_dataset(branch, data=np_data)
            except ValueError:
                # If conversion to numpy is not possible, store as awkward array
                tree_group.create_dataset(branch, data=ak.to_buffers(branch_data))
                
            print(f"Processed branch: {branch}")
    h5_files_path = 'XXX'
    print(f"Conversion completed. HDF5 file created: {h5_files_path}")



# def create_dataframe_from_hdf5_scenario_2(h5_file_path: str) -> None:
#     df_rows = []
#     with h5py.File(h5_file_path, 'r') as h5_file:
#         print(f"Creating a dataframe for: {h5_file_path}")
#         for column in h5_file.keys():
#             if isinstance(h5_file[column], h5py.Group):
#                 print(f"Column: {column} is a group and contains {len(h5_file[column])} datasets:") 
#                 for subkey in h5_file[column].keys():
#                     data = h5_file[column][subkey][:]
#                     # row = {h5_file[column][subkey][:]}
#                     # print(f"Dataset_subkey: {subkey}")
#                     row = {f'{column}': data} ##BUG : IS ORDERING NOT PRESERVED ?
#                     df_rows.append(row)
#                     # print(data)


#             else:
#                 print(f"Column: {column} is NOT a group and contains {len(h5_file[column])} datasets:")
#                 data = h5_file[column][:]
#                 print(f"Dataset: {column}")
#                 # print(data)
#                 df_rows.append(data)

#     return df_rows


def create_dataframe_from_hdf5_scenario_3(h5_file_path: str) -> pd.DataFrame:

    total_data = {}

    # Open the H5 file
    with h5py.File(h5_file_path, 'r') as h5_file:
        # List all groups (datasets)
        keys = list(h5_file.keys())
        print("Keys in the H5 file:", keys)

        # Iterate over all datasets and read their contents

        for key in keys:
            data = h5_file[key][:]
            total_data[key] = data
            # Print the data
            # print(f"Data in the '{key}' dataset:", data)
    print(f'data for {h5_file_path}:')

    return pd.DataFrame(total_data)


def scan_for_new_root_files(root_dir: str, h5_dir:str=None, sqlite_dir:str=None) -> List[str]:
    """
    Scans the root directory for new files that have not been converted to h5 AND sqlite files.

    Parameters:
     - root_dir (str): The directory containing the ROOT files.
     - h5_dir (str): The directory containing the HDF5 files.
     - sqlite_dir (str): The directory containing the SQLite files.
    
    Returns:
     - new_root_files (list): A list of file paths to the new ROOT files.
    
    """
    root_files = os.listdir(root_dir)
    h5_files = [x.split('.h5')[0] for x in os.listdir(h5_dir)]
    sqlite_files = [x.split('.sqlite3')[0] for x in os.listdir(sqlite_dir)]
    # h5
    new_root_for_h5 = [x for x in root_files if x.endswith('.root') and x.split('.root')[0] not in h5_files]
    if not new_root_for_h5:
        print('No new root files found for h5')
    else:
        print(f'New root files: {new_root_for_h5} for h5')
    # sqlite
    new_root_for_sqlite = [x for x in root_files if x.endswith('.root') and x.split('.root')[0] not in sqlite_files]
    if not new_root_for_sqlite:
        print('No new root files found for sqlite')
    else:
        print(f'New root files: {new_root_for_sqlite} for sqlite')

    new_root_files_h5 = [os.path.join(os.getcwd(), 'data', 'root', x) for x in new_root_for_h5]
    new_root_files_sqlite = [os.path.join(os.getcwd(), 'data', 'root', x) for x in new_root_for_sqlite]

    return new_root_files_h5, new_root_files_sqlite


def convert_new_root_files(root_files: List[str], h5_dir:str) -> None:
    """
    Converts a list of root files to HDF5 format and saves them in the specified directory.
    - The function converts each root file in the `root_files` list to HDF5 format.
    - The converted HDF5 files are saved in the `h5_dir` directory.
    - The function uses the `root_to_arrays()` function to convert the root files to arrays.
    - The converted arrays are then saved to HDF5 files using the `save_to_h5()` function.

    Parameters:
     - root_files (List[str]): A list of root file paths to be converted.
     - h5_dir (str): The directory where the converted HDF5 files will be saved.

    Returns:
        None

    Examples:
        >>> root_files = ['/path/to/file1.root', '/path/to/file2.root']
        >>> h5_dir = '/path/to/h5_files'
        >>> convert_new_root_files(root_files, h5_dir)
    """
    for file in root_files:
        columns_to_find = ["eventNumber", "digitX", "digitY", "digitZ"]
        h5_file_path = os.path.join(h5_dir, os.path.basename(file).replace('.root', '.h5'))
        array_data_dict = root_to_dict_of_arrays(file, columns_to_find)
        awkward_array = root_to_awkward_arrays(file, columns_to_find)
        #FIXME  : create a function to handle the conversion
        save_to_h5(array_data_dict, awkward_array, h5_file_path) 
          
    return None


def list_h5_files(h5_dir: str) -> List[str]:
    """
    Lists all HDF5 files in the given directory.

    Parameters:
     - h5_dir (str): Directory to search for HDF5 files.

    Returns:
     - List[str]: List of HDF5 file paths.
    """
    h5_files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]
    return h5_files


def main():
    root_dir = os.getcwd() + "/data/root"  # Directory containing ROOT files
    output_dir = os.getcwd() + "/data/h5"  # Directory to save HDF5 files

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while True:
        print("Please choose an option:")
        print("1. Read an HDF5 file")
        print("2. Convert ROOT files to HDF5 and save them")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            # List all available HDF5 files
            h5_files = list_h5_files(output_dir)

            if not h5_files:
                print(f"No HDF5 files found in {output_dir}.")
            else:
                print("Available HDF5 files:")
                for idx, file in enumerate(h5_files):
                    print(f"{idx + 1}. {file}")

                file_choice = int(input(f"Select a file to read (1-{len(h5_files)}): "))
                selected_h5_file = os.path.join(output_dir, h5_files[file_choice - 1])

                read_h5_file(selected_h5_file)
                df = create_dataframe_from_hdf5_scenario_3(selected_h5_file)
                print(df)

        elif choice == '2':
            print(f"Scanning {root_dir} for new ROOT files to convert...")
            new_root_files_h5, new_root_files_sqlite = scan_for_new_root_files(root_dir, output_dir, sqlite_dir=None)

            if not new_root_files_h5:
                print("No new ROOT files found for conversion.")
            else:
                print(f"Converting {len(new_root_files_h5)} ROOT files to HDF5 format...")
                convert_new_root_files(new_root_files_h5, output_dir)
                print(f"Conversion completed. HDF5 files saved to {output_dir}")

        elif choice == '3':
            print("Exiting...")
            break

        else:
            print("Invalid option, please choose again.")

if __name__ == "__main__":
    main()

