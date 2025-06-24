import logging

import time
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from typing import Optional
import numpy as np
import pandas as pd
import h5py



class SWMR_HDF5Handler:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.ensure_swmr_hdf5_file()

    def ensure_swmr_hdf5_file(self):
        directory = os.path.dirname(self.file_path)
        if not os.path.exists(directory):
            logging.info(f"[SWMR] Directory does not exist. Creating: {directory}")
            os.makedirs(directory, exist_ok=True)

        if not os.path.exists(self.file_path):
            logging.info(f"[SWMR] HDF5 file not found. Creating: {self.file_path}")
            try:
                with h5py.File(self.file_path, 'w', libver='latest') as f:
                    f.swmr_mode = True  # 💡 Mark SWMR-ready immediately
                logging.info(f"[SWMR] Created new HDF5 file with SWMR support: {self.file_path}")
            except Exception as e:
                logging.error(f"[SWMR] Failed to create HDF5 file: {e}", exc_info=True)
        else:
            logging.debug1(f"[SWMR] HDF5 file already exists: {self.file_path}")





    def load_image(self, dataset_path: str) -> Optional[np.ndarray]:
        logging.debug1(f"[SWMR] Reading from: {self.file_path} [{dataset_path}]")
        try:
            with h5py.File(self.file_path, 'r', swmr=True) as f:
                return f[dataset_path][()]
        except KeyError:
            logging.warning(f"[SWMR] Dataset path not found: {dataset_path}")
            return None
        except Exception as e:
            logging.error(f"[SWMR] Failed to load image: {e}", exc_info=True)
            return None


    def store_image(
        self,
        dataset_path: str,
        image_data: np.ndarray,
        attributes: Optional[dict] = None,
        attribute_process: str = "att_replace",
    ):
        attributes = attributes or {}
        logging.debug1(f"[SWMR] Writing to {self.file_path} [{dataset_path}]")

        # retry-with-back-off helps when another writer has the lock
        for attempt in range(5):
            try:
                with h5py.File(self.file_path, "a", libver="latest") as f:
                    self.handle_dataset(
                        hdf5_file=f,
                        dataset_name=dataset_path,
                        numpy_array=image_data,
                        attributes_new=attributes,
                        attribute_process=attribute_process,
                    )
                    f.flush()           # <-- important for SWMR readers
                break                   # success → leave retry loop
            except OSError as e:
                if getattr(e, "errno", None) == 11 and attempt < 4:
                    time.sleep(0.25)   # file is temporarily locked
                    continue
                logging.error(f"[SWMR] Error storing image: {e}", exc_info=True)
                raise




    @staticmethod
    def handle_dataset(hdf5_file, dataset_name, numpy_array, attributes_new, attribute_process):
        if dataset_name in hdf5_file:
            existing_dataset = hdf5_file[dataset_name]
            existing_attrs = dict(existing_dataset.attrs)

            if attribute_process == "att_replace":
                merged_attrs = {**existing_attrs, **attributes_new}
            elif attribute_process == "att_delete":
                merged_attrs = attributes_new
            elif attribute_process == "att_append":
                merged_attrs = {
                    **existing_attrs,
                    **{k: [existing_attrs.get(k), attributes_new[k]] for k in attributes_new if k in existing_attrs}
                }
            else:
                merged_attrs = attributes_new

            del hdf5_file[dataset_name]
        else:
            merged_attrs = attributes_new

        group_path = os.path.dirname(dataset_name)
        if group_path and group_path not in hdf5_file:
            hdf5_file.require_group(group_path)

        dset = hdf5_file.create_dataset(dataset_name, data=numpy_array)
        sanitized = SWMR_HDF5Handler.sanitize_hdf5_attrs(merged_attrs)
        for key, value in sanitized.items():
            try:
                dset.attrs[key] = value
            except Exception as e:
                logging.error(f"Failed to store attr: {key} = {value} ({type(value)}): {e}")


    @staticmethod
    def sanitize_hdf5_attrs(attr_dict):
        sanitized = {}
        for k, v in attr_dict.items():
            if v is None:
                continue  # skip None values entirely
            elif isinstance(v, str):
                sanitized[k] = v.encode("utf-8")  # h5py requires bytes
            elif isinstance(v, (int, float, np.integer, np.floating)):
                sanitized[k] = v
            elif isinstance(v, (list, tuple)):
                try:
                    sanitized[k] = np.array(v)
                except Exception:
                    logging.warning(f"Could not convert list attr {k}: {v}")
            else:
                logging.warning(f"Unsupported attr type: {k} = {v} ({type(v)}) — Skipping")
        return sanitized