"""dataset from https://singlecell.broadinstitute.org/single_cell/study/SCP1830/spatial-atlas-of-molecular-cell-types-and-aav-accessibility-across-the-whole-mouse-brain?utm_source=chatgpt.com#study-download"""
"""IO functions for STARmap technology."""
import os

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from ..configuration import SKM
from ..logging import logger_manager as lm
from .utils import get_points_props

def starmap_plus(
    folder_path:str
) -> AnnData:
    """ Read StarMAP Plus output files and construct an AnnData object.
   
    Parameters:
    -----------
        folder_path:str
            Path to the folder containing STARmap Plus data files.
            The folder must include three files whose filenames contain the following keywords:
            - "spatial" --------> CSV file with spatial coordinates (e.g., well_2_5_spatial.csv)
            - "spot_meta"-------> CSV file with spot or cell metadata (e.g., well_2_5_spot_meta.csv)
            - "raw_expression_pd"------> CSV file with gene expression matrix (e.g., well_2_5processed_expression_pd.csv)
    Returns:
    --------
        AnnData object containing the STARmap Plus data.
        
    Raises:
    ------
    FileNotFoundError
        If any of the required files are not found in the folder.
    """
    
    files = os.listdir(folder_path)
    
    def find_file(keyword):
        matches = [f for f in files if keyword in f]
        if not matches:
            raise FileNotFoundError(f"No file with keyword '{keyword}' found in {folder_path}")
        return os.path.join(folder_path, matches[0])
    
    # Automatically locate each required file
    spatial_path = find_file("spatial")
    meta_path = find_file("spot_meta")
    expression_path = find_file("raw_expression_pd")

    expr_df = pd.read_csv(expression_path)
    expr_df = expr_df.rename(columns={"GENE": "gene"}).set_index("gene")
    # transpose to spot × gene , index --> spot
    expr_df = expr_df.T  

    spatial_df = pd.read_csv(spatial_path, skiprows=[1], index_col=0)
    spot_meta = pd.read_csv(meta_path, index_col=0)
    
    lm.main_info("Constructing count matrix.")
    adata = AnnData(X=expr_df.values)
    adata.obs_names = expr_df.index
    adata.var_names = expr_df.columns

    adata.obsm["spatial"] = spatial_df[["X", "Y", "Z"]].values

    metadata_cols = [
        col for col in spatial_df.columns 
        if col not in ["X", "Y", "Z"]
    ]
    adata.obs = spatial_df[metadata_cols]

    #adata.uns["spot_meta"] = spot_meta.to_dict(orient="list")
    adata.uns["spot_meta"] = spot_meta

    scale, scale_unit = 1.0, None

    # Set uns
    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    SKM.init_uns_pp_namespace(adata)
    SKM.init_uns_spatial_namespace(adata)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, scale)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, scale_unit)

    return adata
