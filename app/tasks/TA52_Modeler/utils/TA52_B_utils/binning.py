import logging
import re
from typing import Dict, Any
import numpy as np



def binning(job):
    """
    Performs feature binning on the job's numeric data matrix (`job.attrs.data_train`) using
    the binning configuration defined in `job.input.metricModel_instructions.binning_cfg`.

    Supports both explicit and implicit binning strategies:
    - 'explicit': groups predefined feature sets under labels.
    - 'implicit': regex-based pattern matching to dynamically define bins.

    Additionally:
    - Assigns an 'index' bin if specified by the blacklist.
    - Computes a 'rest' bin for unassigned features.
    
    Results are stored in:
        job.attrs.bin_dict → Dict[str, Dict] with:
            {
                "X":      np.ndarray of selected columns,
                "input_cols": list of original column names,
                "dest_prefix": str label for column prefixing
            }

    Parameters:
    ----------
    job : ModelerJob
        The full job object containing input config and numeric data matrix.

    Returns:
    -------
    job : ModelerJob
        The updated job with `.attrs.bin_dict` populated.
    
    Raises:
    ------
    ValueError:
        If required config elements (e.g. cluster definitions or patterns) are missing.
    """



    X = job.attrs.data_train  # assumed to be numpy or cupy ndarray
    bin_cfg = job.input.metricModel_instructions.binning_cfg
    name2idx = job.attrs.encoder.cols                   # {col_name: index}
    idx2name = {v: k for k, v in name2idx.items()}      # {index: col_name}
    all_col_names = list(name2idx.keys())               # list of column names in order
    assigned_cols = set()
    out = {}

    if bin_cfg.strategy == "explicit":
        exp = bin_cfg.explicit
        if not exp or not exp.clusters:
            raise ValueError("Explicit binning requires non-empty 'clusters'.")

        presets = exp.presets or {}
        for label, members in exp.clusters.items():
            col_names = []
            for m in members:
                col_names += presets[m] if m in presets else [m]
            idx = [name2idx[c] for c in col_names if c in name2idx]
            bin_X = X[:, idx]
            assigned_cols.update(col_names)
            out[label] = {
                "X": bin_X,
                "input_cols": col_names,
                "dest_prefix": label
            }

            #logging.debug2(
            #    f"[BINNING][{label}] shape={bin_X.shape} | "
            #    f"cols=[{', '.join(col_names[:5])}" +
            #    (" ..." if len(col_names) > 8 else "") +
            #    f" {', '.join(col_names[-3:])}]"
            #)

    elif bin_cfg.strategy == "implicit":
        imp = bin_cfg.implicit
        if not imp or not imp.patterns:
            raise ValueError("Implicit binning requires non-empty 'patterns'.")

        flags = re.I if imp.ignore_case else 0
        for label, subs in imp.patterns.items():
            pat = re.compile("|".join(map(re.escape, subs)), flags)
            idx = [i for i, n in idx2name.items() if pat.search(n)]
            if not idx:
                logging.warning(f"[BINNING] Bin '{label}' matched 0 columns.")
                continue

            input_cols = [idx2name[i] for i in idx]
            bin_X = X[:, idx]
            assigned_cols.update(input_cols)
            out[label] = {
                "X": bin_X,
                "input_cols": input_cols,
                "dest_prefix": label
            }

            #logging.debug2(
            #    f"[BINNING][{label}] shape={bin_X.shape} | "
            #    f"cols=[{', '.join(input_cols[:5])}" +
            #    (" ..." if len(input_cols) > 8 else "") +
            #    f" {', '.join(input_cols[-3:])}]"
            #)

    else:
        raise ValueError(f"Unsupported binning strategy: {bin_cfg.strategy}")

    # Optional index bin
    index_cols = job.attrs.blacklist.get("index_cols", [])
    if index_cols:
        idx = [name2idx[c] for c in index_cols if c in name2idx]
        bin_X = X[:, idx]
        out["index"] = {
            "X": bin_X,
            "input_cols": index_cols,
            "dest_prefix": "index"
        }

        #logging.debug2(
        #    f"[BINNING][index] shape={bin_X.shape} | cols={index_cols}"
        #)

    # "rest" bin (unassigned columns)

    unassigned_cols = sorted(set(all_col_names) - set(assigned_cols) - set(index_cols))
    if unassigned_cols:
        rest_idx = [name2idx[c] for c in unassigned_cols]
        bin_X = X[:, rest_idx]
        out["rest"] = {
            "X": bin_X,
            "input_cols": unassigned_cols,
            "dest_prefix": "rest"
        }

        logging.debug2(
            f"[BINNING][rest] shape={bin_X.shape} | "
            f"cols=[{', '.join(unassigned_cols[:5])}" +
            (" ..." if len(unassigned_cols) > 8 else "") +
            f" {', '.join(unassigned_cols[-3:])}]"
        )

    job.attrs.bin_dict = out
    return job
