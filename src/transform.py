import uproot
import h5py
import os
import numpy as np
import pandas as pd
import awkward as ak #pip install akward-pandas 
from typing import List

def root_to_h5(root_files_path: str, h5_file: str) -> None:
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

    # Create an HDF5 file
    # with h5py.File(h5_files_path, 'w') as h5_file:
        # Iterate through trees in the ROOT file
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



def get_tree_columns(root_file: str, cols_to_find: List[str]) -> dict:
    """
    Extracts and returns the columns of interest from a ROOT file. It identifies trees within the ROOT file 
    that contain all specified columns and returns a dictionary mapping tree names to the list of these columns.

    * Parameters:
        root_file_path (str): The file path to the ROOT file.

    * Returns:
        tree_dict: A dictionary where the keys are tree names and the values are lists of columns found in those trees.
    """
    tree_dict = {}
    for tree_name in root_file.keys():
        tree = root_file[tree_name]
        if all(col in tree.keys() for col in cols_to_find):
            tree_list = [col for col in cols_to_find if col in tree.keys()]
            tree_dict[tree_name] = tree_list
            print(f"Found columns: {tree_list} in tree: {tree_name}")
        else:
            print(f"Columns not found in tree: {tree_name}")
    return tree_dict

def scan_for_new_root_files(root_dir: str, h5_dir:str) -> list:
    root_files = os.listdir(root_dir)
    # h5_files = os.listdir(h5_dir)
    h5_files = [x.split('.h5')[0] for x in os.listdir(h5_dir)]
    new_root = [x for x in root_files if x.endswith('.root') and x.split('.root')[0] not in h5_files]
    print(f'New root files: {new_root}')
    new_root_files = [os.path.join(os.getcwd(), 'root', x) for x in new_root]
    return new_root_files

def convert_new_root_files(root_files: list, h5_dir:str) -> None:
    if not root_files:
        print("No new root files found")
        return
    else:
        for file in root_files:
            h5_file_path = os.path.join(h5_dir, os.path.basename(file).replace('.root', '.h5'))
            # (h5_dir, file.split('.root')[0]+'.h5')
            #  os.path.basename(file).replace('.root', '.h5')
            # h5_file_path = os.path.join(h5_dir, f"{base_name}.h5")
            with h5py.File(h5_file_path, 'w') as h5_file_name:
                h5_file = h5_file_name
                root_to_h5(file, h5_file) 

            # print(f"New empty HDF5 file created: {h5_file_path}")
            ## TODO : convert root to h5 <!>            
    return None

if __name__ == "__main__":

    # cols_to_find = ['eventNumber', 'digitX', 'digitY', 'digitZ', 'digitT']

    # # check data from .root file
    # cols_to_find = ['eventNumber', 'digitX', 'digitY', 'digitZ', 'digitT']
    # dt = pd.DataFrame(columns=cols_to_find)
    # root_files = os.listdir(os.getcwd()+'/data/root')
    # print(root_files)
    # for file in root_files:
    #     root_file_path = f"data/root/{file}"
    #     root_file = uproot.open(root_file_path)   
    #     tree_dict = get_tree_columns(root_file, cols_to_find)
    #     # print(file, tree_dict)
    #     for tree_name in tree_dict.keys():
    #     # print(f"\nProcessing tree: {tree_name}")
    #         df = root_file[tree_name].arrays(library="pd")
    #         print(f"Shape of the DataFrame: {df.shape}")
    #         dt = pd.concat([dt, df[cols_to_find]])
    # # print(dt)
    # print(tree_dict)
    # print(dt.shape)
    # print(len(dt.eventNumber.unique()))


    
    root_dir = f"{os.getcwd()}/data/root/"
    h5_dir = f"{os.getcwd()}/data/h5/"

    # tree_dict = get_tree_columns(root_file, cols_to_find)
    files = scan_for_new_root_files(root_dir, h5_dir)
    print(files)
    convert_new_root_files(files, h5_dir)
    
    ## scan root folder and convert all files to h5
    # root_files = os.listdir(root_files_path)
    # for file in root_files:

    #     root_file_path = f"{root_files_path}{file}"   
    #     h5_file_path = f"{h5_files_path}{file.replace('.root', '.h5')}"
    #     root_file = uproot.open(root_file_path)
    #     tree_dict = get_tree_columns(root_file, cols_to_find)

    #     h5 = root_to_h5(root_file_path, h5_file_path)
    #     print(h5)
    #     print(f"Conversion completed. HDF5 file created: {h5_file_path}")
    #     print(f"Conversion completed. HDF5 file created: {h5_files_path}")
    #     print(f"Conversion completed. HDF5 file created: {h5_files_path}{file.replace('.root', '.h5')}")
    #     print(f"Conversion completed. HDF5 file created: {h5_files_path}{file.replace('.root', '.h5')}")

    # print(f"Conversion completed. HDF5 file created: {h5_files_path}")
