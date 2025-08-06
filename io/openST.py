"""IO functions for openST technology."""
"""dataset from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7990097"""
import os
import h5py

import numpy as np
import pandas as pd
from anndata import AnnData
from anndata._io.specs import read_elem
from scipy.sparse import csr_matrix

from ..configuration import SKM
from ..logging import logger_manager as lm
from .utils import get_points_props

def openST(
    path: str, 
) -> AnnData:
    """
    Load an Open-ST .h5ad file and return an AnnData object.
    Parameters:
    -------------------------------------------------------
        path (str): File path to the .h5 file.

    Returns:
    --------------------------------------------------------
        AnnData object containing the openST data.
    """

    with h5py.File(path, "r") as f:
        X = read_elem(f["X"])
        obs = pd.DataFrame(read_elem(f["obs"]))
        var = pd.DataFrame(read_elem(f["var"]))
        obsm = {}

        if "obsm" in f:
            for key in f["obsm"]:
                obsm[key] = read_elem(f["obsm"][key])

    """create AnnData object"""
    adata = AnnData(
                X = csr_matrix(X) if not isinstance(X, np.ndarray) else X,
                obs = obs,
                var = var,
                obsm = obsm
            )

    scale, scale_unit = 1.0, None
 
    # Set uns
    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    SKM.init_uns_pp_namespace(adata)
    SKM.init_uns_spatial_namespace(adata)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, scale)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, scale_unit)

    return adata
