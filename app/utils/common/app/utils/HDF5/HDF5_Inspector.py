from typing import Optional
import numpy as np
import pandas as pd
import h5py
import logging




class HDF5Inspector:

    @staticmethod
    def get_all_dataset_paths(file_path: str) -> list:
        """Return all dataset paths in the given HDF5 file."""
        paths = []

        def visitor(name, node):
            if isinstance(node, h5py.Dataset):
                paths.append(name)

        with h5py.File(file_path, 'r', swmr=True) as f:
            f.visititems(visitor)
        return paths
    

    @staticmethod
    def collect_attributes_for_dataset(file: h5py.File, dataset_path: str) -> dict:
        """
        Collect attributes from dataset path upward, assuming file is already open with swmr=True.
        """
        attrs = {}
        try:
            obj = file[dataset_path]
            attrs["dataset_shape_drop"] = obj.shape

            while obj.name != '/':
                for k, v in obj.attrs.items():
                    if k not in attrs:
                        attrs[k] = v
                obj = obj.parent

        except KeyError:
            logging.warning(f"Path {dataset_path} not found.")
        attrs["path"] = dataset_path
        return attrs

    @staticmethod
    def HDF5_meta_to_df(file_path: str) -> pd.DataFrame:
        """
        Retrieve a DataFrame of dataset paths and attributes using SWMR.
        """
        dataset_paths = HDF5Inspector.get_all_dataset_paths(file_path)
        with h5py.File(file_path, 'r', swmr=True) as f:
            records = [
                HDF5Inspector.collect_attributes_for_dataset(f, path)
                for path in dataset_paths
            ]
        return pd.DataFrame(records)

    @staticmethod
    def collect_attributes_for_dataset_threadSafe(file_path: str, dataset_path: str) -> dict:
        """
        Open file in SWMR mode for thread-safe and process-safe reads.
        """
        attrs = {}
        try:
            with h5py.File(file_path, 'r', swmr=True) as f:
                obj = f[dataset_path]
                attrs["dataset_shape_drop"] = obj.shape
                while obj.name != '/':
                    for k, v in obj.attrs.items():
                        if k not in attrs:
                            attrs[k] = v
                    obj = obj.parent
        except KeyError:
            logging.warning(f"Path {dataset_path} not found.")
        attrs["path"] = dataset_path
        return attrs
