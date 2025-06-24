import logging
import h5py
from typing import Optional, Literal
import numpy as np
import pandas as pd
from pathlib import Path
import os
import subprocess




class HDF5Utils:
      
    
    @staticmethod
    def add_attributes_to_hdf5(hdf5_file_path, attributes):
        def add_attributes(obj):
            for key, value in attributes.items():
                obj.attrs[key] = value
        try:
            logging.debug1(f"Adding attributes to HDF5 file: {hdf5_file_path}")
            with h5py.File(hdf5_file_path, "a") as hdf_file:
                def traverse_and_add_attributes(name, obj):
                    add_attributes(obj)
                hdf_file.visititems(traverse_and_add_attributes)
                logging.debug1("Attributes added successfully.")
                return ["Finished", "Success"]
        except Exception as e:
            logging.error(f"Error: {e}", exc_info=True)
            return ["failed", f"Error: {e}"]
        
    @staticmethod
    def hdf5_to_dict(self, hdf5_object, method: Literal["immediate", "relative"] = "relative"):
        logging.debug1("Converting HDF5 object to dictionary.")
        result = {}
        for key in hdf5_object.keys():
            item = hdf5_object[key]
            if isinstance(item, h5py.Dataset):
                if method == "immediate":
                    logging.debug1(f"Converting dataset: {key}")
                    result[key] = item[()]
                elif method == "relative":
                    logging.debug1(f"Adding dataset path: {key}")
                    result[key] = item.name
            elif isinstance(item, h5py.Group):
                logging.debug1(f"Entering group: {key}")
                result[key] = self.hdf5_to_dict(item, method=method)

            attr_dict = {attr_key: item.attrs[attr_key] for attr_key in item.attrs}
            if attr_dict:
                logging.debug1(f"Adding attributes for: {key}")
                if key not in result:
                    result[key] = {}
                if isinstance(result[key], dict):
                    result[key]['_attributes'] = attr_dict
                else:
                    result[key] = {'_attributes': attr_dict}

        logging.debug1(f"Conversion to dictionary complete. With method: {method}. Stored a total of {len(result)} items.")
        return result

    @staticmethod
    def delete_keys_from_hdf5(hdf5_file_path, keys_to_delete):
        try:
            logging.debug1(f"Deleting specified keys from HDF5 file: {hdf5_file_path}")
            with h5py.File(hdf5_file_path, "a") as hdf_file:
                for key in keys_to_delete:
                    if key in hdf_file:
                        logging.debug1(f"Deleting key: {key}")
                        del hdf_file[key]
                    else:
                        logging.warning(f"Key not found, skipping: {key}")
            logging.debug1("Specified keys deleted successfully.")
            return ["Finished", "Success"]
        except Exception as e:
            logging.error(f"Error: {e}", exc_info=True)
            return ["failed", f"Error: {e}"]

    @staticmethod
    def unlock_dirty_hdf5_files(directories: str):
        logging.info(f"🔍 Scanning for locked HDF5 files in: {directories}")
        
    
    
    
        for dir_path in directories:
            if not os.path.exists(dir_path):
                logging.warning(f"❌ Directory  does not exist: {dir_path}")
                continue

            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(".hdf5"):
                        file_path = os.path.join(root, file)
                        file_path = os.path.abspath(file_path)
                        try:
                            logging.debug2(f"Running h5clear: h5clear -s {file_path}")
                            subprocess.run(["h5clear", "-s", file_path], check=True)
                            logging.info(f"✅ Cleared HDF5 lock: {file_path}")
                        except Exception as e:
                            logging.error(f"⚠️ Failed to clear '{file_path}': {e}")
                            