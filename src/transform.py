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
#      SCENARIO 1 : padding arrays with different lengths to the same length
##########################################################################################
    """
    * Handling digitX: Since digitX contains arrays of different lengths, we pad each array to the length of the longest array using np.pad. This ensures that all arrays have the same length and can be stored in a 2D dataset in the HDF5 file.
    * Padded Data: The padded arrays are stored in the HDF5 file, with np.nan used to fill the padding.

    When storing data from a ROOT file into an HDF5 file, padding is necessary for columns like digitX and digitY that contain arrays of arrays (i.e., nested arrays) with varying lengths. Here's why padding is important:
    Reasons for Padding:
    1.Uniform Shape Requirement:
    DF5 datasets require that all elements have the same shape. If digitX and digitY contain arrays of different lengths, they cannot be directly stored in a single HDF5 dataset without making their lengths uniform.
    2.Data Integrity:
    Padding ensures that the structure of the data is preserved when converting from ROOT to HDF5. Without padding, you might lose information about the original structure of the nested arrays.
    3.Ease of Access:
    By padding the arrays to the same length, you can easily access and manipulate the data in a consistent manner. This uniformity simplifies downstream data processing and analysis.

    <!> Check for Nested Arrays: The code checks if the column's data type is object ('O'), which indicates nested arrays.
    """
    # with h5py.File(h5_file_path, 'w') as hdf5_file:
    #         # Step 4: Store the fetched data
    #         for key_, values_ in arrays.items():
    #             if arrays[key_].dtype == 'O':
    #                 # Handle the array of arrays case
    #                 max_len = max(len(arr) for arr in values_)
    #                 padded_data = np.array([np.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=np.nan) for arr in values_])
    #                 hdf5_file.create_dataset(key_, data=padded_data)
    #             else:
    #                 hdf5_file.create_dataset(key_, data=values_)

##########################################################################################
#      SCENARIO 2 : store each subarray as a separate dataset within the group
##########################################################################################
    # if arrays is an instance of ak.Array
    # if isinstance(awkward_arrays, ak.Array): #isinstance(ak.type(awkward_arrays), ak.types.ListType)
    #     with h5py.File(h5_file_path, 'w') as h5_file: 
    #         # Example: awkward_arrays['digitX']
    #         # ArrayType(ListType(NumpyType('float64')), 2108, None)
    #         # <Array [[-15.2, -47.8, ..., 26.9, -54.9], ...] type='2108 * var * float64'>
    #         for field in awkward_arrays.fields:
    #             array_type = ak.type(awkward_arrays[field])
    #             if isinstance(array_type, ak.types.ArrayType) and \
    #             isinstance(array_type.content, ak.types.ListType) and \
    #             isinstance(array_type.content.content, ak.types.NumpyType):  
    #         #    ak.type(awkward_arrays[field]).type.type.dtype == 'float64' and \
    #         #    ak.type(awkward_arrays[field]).parameters == 2108:
    #             # if 
    #             # group for each field
    #                 group = h5_file.create_group(field)
    #                 # each subarray as a separate dataset within the group
    #                 for i, subarray in enumerate(awkward_arrays[field]):
    #                     group.create_dataset(str(i), data=ak.to_numpy(subarray))
    #             else:
    #                 h5_file.create_dataset(field, data=ak.to_numpy(arrays[field]))
    # return None

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
                # Convert object dtype arrays to fixed-length strings
                # max_length = max(len(str(item)) for item in array)
                # str_dtype = h5py.string_dtype(encoding='utf-8', length=max_length)
                # array = array.astype(str_dtype)
            # h5_file.create_dataset(key, data=array)


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
    # tree_dict = get_tree_branches(root_file, columns_to_find) # get a dictionary with the tree and columns(branches) to extract
    # for tree_name, cols in tree_dict.items():
    #     arrays = root_file[tree_name].arrays(cols, library="np")
    # return arrays

    # for tree_name in tree_dict.keys():
    #     root_file[tree_name].arrays(library="np")['eventNumber'] # method 1
    #     root_file[tree_name].arrays(columns_to_find)['eventNumber']# method 2
        # print(f"\nProcessing tree: {tree_name}")


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



def create_dataframe_from_hdf5_scenario_2(h5_file_path: str) -> None:
    df_rows = []
    with h5py.File(h5_file_path, 'r') as h5_file:
        print(f"Creating a dataframe for: {h5_file_path}")
        for column in h5_file.keys():
            if isinstance(h5_file[column], h5py.Group):
                print(f"Column: {column} is a group and contains {len(h5_file[column])} datasets:") 
                for subkey in h5_file[column].keys():
                    data = h5_file[column][subkey][:]
                    # row = {h5_file[column][subkey][:]}
                    # print(f"Dataset_subkey: {subkey}")
                    row = {f'{column}': data} ##BUG : IS ORDERING NOT PRESERVED ?
                    df_rows.append(row)
                    # print(data)


            else:
                print(f"Column: {column} is NOT a group and contains {len(h5_file[column])} datasets:")
                data = h5_file[column][:]
                print(f"Dataset: {column}")
                # print(data)
                df_rows.append(data)

    return df_rows


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



# def scan_for_not_converted_files(root_dir: str, h5_dir:str=None, sqlite_dir:str=None) -> List[str]:
#     """
#     Scans the root directory for new files that have not been converted to h5 or sqlite files
#     depending on the folder specified.

#     Parameters:
#      - root_dir (str): The directory containing the ROOT files.
#      - h5_dir (str): The directory containing the HDF5 files.
#      - sqlite_dir (str): The directory containing the SQLite files.
    
#     Returns:
#      - new_root_files (list): A list of file paths to the new ROOT files.
    
#     """
#     root_files = os.listdir(root_dir)
#     dir_to_check = input('check for h5 or sqlite files: ')
#     if dir_to_check == 'h5':
#         files = [x.split('.h5')[0] for x in os.listdir(h5_dir)]
#     elif dir_to_check == 'sqlite':
#         files = [x.split('.sqlite3')[0] for x in os.listdir(sqlite_dir)]
#     else:
#         print('Invalid directory name')
#         return None
#     # check for files in root folder that are not in h5 and sqlite folders 
#     new_root = [x for x in root_files if x.endswith('.root') and x.split('.root')[0] not in files]
    
#     print(f'New root files: {new_root}')
#     print(f'Searched in {dir_to_check} folder')
#     new_root_files = [os.path.join(os.getcwd(), 'root', x) for x in new_root]
    
#     return new_root_files


# same with the next but with no DOC
# def convert_new_root_files(root_files: List[str], h5_dir:str) -> None:
#     """
    
#     """
#     # if not root_files:
#     #     print("No new root files found")
#     #     return
#     # else:
#     for file in root_files:
#         h5_file_path = os.path.join(h5_dir, os.path.basename(file).replace('.root', '.h5'))
#         array_data_dict = root_to_arrays(file, columns_to_find)
#         save_to_h5(array_data_dict, h5_file_path)     
            
#             #  os.path.basename(file).replace('.root', '.h5')
#             # root_to_h5(file, cols_to_find, h5_file_path)

#             # with h5py.File(h5_file_path, 'w') as f:
#                 # h5_file = h5_file_name
#                 # root_to_h5(file, f) 

#             # print(f"New empty HDF5 file created: {h5_file_path}")
#             ## TODO : convert root to h5 <!>            
#     return None


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
        h5_file_path = os.path.join(h5_dir, os.path.basename(file).replace('.root', '.h5'))
        array_data_dict = root_to_dict_of_arrays(file, columns_to_find)
        awkward_array = root_to_awkward_arrays(file, columns_to_find)
        #FIXME  : create a function to handle the conversion
        save_to_h5(array_data_dict, awkward_array, h5_file_path) 
          
    return None


if __name__ == "__main__":
    try:
    # STEP 1: convert ROOT to H5
    # initialize variables
        root_dir = f"{os.getcwd()}/data/root/"
        h5_dir = f"{os.getcwd()}/data/h5/"
        sqlite_dir = f"{os.getcwd()}/data/sqlite/"

        columns_to_find =['eventNumber', 'digitX', 'digitY', 'digitZ', 'digitT']

        files_path = scan_for_new_root_files(root_dir, h5_dir, sqlite_dir)
        # dhf5 conversion
        try:
            convert_new_root_files(files_path[0], h5_dir)
        except Exception as error_hdf5:
            print(error_hdf5)
        # sqlite conversion
        # try:
        #     convert_new_root_files(files_path[1], sqlite_dir)
        # except Exception as error_sqlite:
        #     print(error_sqlite)

    # STEP 2: create a dataframe from read H5 file  
        data = create_dataframe_from_hdf5_scenario_3('/home/nikosili/projects/annie_gnn/data/h5/after_phase_0.9.h5')
        print(pd.DataFrame(data))

    except Exception as error_main:
        print(error_main)