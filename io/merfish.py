"""IO functions for MERFISH technology."""
"""dataset from https://github.com/lhqing/merfishing/tree/main/docs/user_guide/dummy_experiment/output/region_0"""
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix,issparse

from ..configuration import SKM
from ..logging import logger_manager as lm
def read_merfish(
    path: str,
    meta_path: str,
) -> AnnData:
    """Read MERFISH data as AnnData.

    Args:
        path: Path to matrix files
        positions_path: Path to csv containing spatial coordinates
    """

    lm.main_info("Constructing count matrix.")
    X = pd.read_csv(path, index_col=0)
    obs = pd.DataFrame(index=X.index)
    var = pd.DataFrame(index=X.columns)

    adata = AnnData(X=csr_matrix(X, dtype=np.uint16), obs=obs, var=var)

    df_loc = pd.read_csv(meta_path, index_col=0, usecols=[0, 1, 3, 4])
   
    df_loc.columns = ["fov", "x", "y"]
    df_loc = df_loc.astype({"fov": np.int16, "x": np.float32, "y": np.float32})
    
    offset = min(df_loc["x"].min(), df_loc["y"].min())
    df_loc["x"] = df_loc["x"] - offset
    df_loc["y"] = df_loc["y"] - offset

    common_ids = np.intersect1d(df_loc.index, adata.obs_names)
    adata = adata[common_ids, :]

    adata.obs["fov"] = df_loc.loc[adata.obs.index,"fov"]
    adata.obsm["spatial"] = np.array(df_loc)

    """simpfy cell_id"""
    simple_ids = [f"cell_{i}" for i in range(1, adata.n_obs + 1)]
    adata.obs["original_cell_id"] = adata.obs.index
    adata.obs.index = simple_ids
    adata._sanitize()

    scale, scale_unit = 1.0, None

    # Set uns
    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    SKM.init_uns_pp_namespace(adata)
    SKM.init_uns_spatial_namespace(adata)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, scale)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, scale_unit)
    return adata
