"""Reader class for spateo read function"""
import os
import cv2
import h5py
import gzip
import math
import scipy.io
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData
from anndata._io.specs import read_elem
from typing_extensions import Literal
from scipy.sparse import csr_matrix
from scipy.sparse import csr_matrix
from typing import List, Optional, Tuple, Union, Callable, Dict, NamedTuple, Optional

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import skimage.io

from ..configuration import SKM
from ..logging import logger_manager as lm
from .utils import (
    bin_indices,
    centroids,
    get_bin_props,
    get_coords_labels,
    get_label_props,
    get_points_props,
    in_concave_hull,
)

try:
    import ngs_tools as ngs

    VERSIONS = {
        "slide2": ngs.chemistry.get_chemistry("Slide-seqV2").resolution,
        "visium": ngs.chemistry.get_chemistry("Visium").resolution,
        "stereo": ngs.chemistry.get_chemistry("Stereo-seq").resolution,
    }
except ModuleNotFoundError:

    from typing import NamedTuple, Optional, Literal

    class SpatialResolution(NamedTuple):
        scale: float = 1.0
        unit: Optional[Literal["nm", "um", "mm"]] = None

    VERSIONS = {
        "slide2": SpatialResolution(10.0, "um"),
        "visium": SpatialResolution(55.0, "um"),
        "stereo": SpatialResolution(0.5, "um")
    }

    COUNT_COLUMN_MAPPING = {
        SKM.X_LAYER: 3,
        SKM.SPLICED_LAYER_KEY: 4,
        SKM.UNSPLICED_LAYER_KEY: 5,
    }


class Reader:
    def __init__(self,tech:str):
        self.tech = tech.lower()
        self.method_map: Dict[str, Callable] = {
            "merfish": self.read_merfish,
            "seqfish": self.read_seqfish,
            "slideseq": self.read_slideseq,
            "stereo": self.read_bgi,
            "starmap": self.read_starmap,
            "visium": self.read_visium,
            "openst": self.read_openST,
        }
    
        if self.tech not in self.method_map:
            raise ValueError(f"Unsupported technology: {self.tech}. Supported types are: {list(self.method_map.keys())}")
        

    """Call the corresponding read function and pass in the parameters required by the technology"""    
    def read(self, *args, **kwargs):
        return self.method_map[self.tech](*args, **kwargs)
    
    #dataset from https://github.com/lhqing/merfishing/tree/main/docs/user_guide/dummy_experiment/output/region_0"""
    # __  __ _____ ____  _____ ___ ____  _   _ 
    #|  \/  | ____|  _ \|  ___|_ _/ ___|| | | |
    #| |\/| |  _| | |_) | |_   | |\___ \| |_| |
    #| |  | | |___|  _ <|  _|  | | ___) |  _  |
    #|_|  |_|_____|_| \_\_|   |___|____/|_| |_|
                                           
    def merfish(
        self,
        path: str,
        meta_path: str,
        ) -> AnnData:
        """Read MERFISH data as AnnData.

        Parameters:
        -------------------------------------------------------------
        path: Path to matrix files
        positions_path: Path to csv containing spatial coordinates

        Returns:
        --------------------------------------------------------------
        AnnData object containing the merfish data.
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
    

   #dataset from https://github.com/drieslab/spatial-datasets/tree/master/data/2019_seqfish_plus_SScortex/
   # ____  _____ ___  _____ ___ ____  _   _ 
   #/ ___|| ____/ _ \|  ___|_ _/ ___|| | | |
   #\___ \|  _|| | | | |_   | |\___ \| |_| |
   # ___) | |__| |_| |  _|  | | ___) |  _  |
   #|____/|_____\__\_\_|   |___|____/|_| |_|

    def seqfish(
        self,
        path: str,
        meta_path: str,
        fov_path: str | None = None,
        accumulate_x: bool = False,
        accumulate_y: bool = False,
    ) -> AnnData:
        """Read seqFISH data as AnnData.

        Parameters:
        ---------------------------------------------------------------------------------
        path: Path to seqFISH digital expression matrix CSV.
        meta_path: Path to CSV file containing cell centroid locations.
        fov_offset: a dataframe contain offset of each fov, for example,
            {'fov':[fov_1, ..], 'x_offset':[x_offset_1, ..], 'y_offset':[y_offset_1, ..]}
        accumulate_x: whether to accumulate x_offset
        accumulate_y: whether to accumulate y_offset
    
        Returns:
        ----------------------------------------------------------------------------------
        AnnData object containing the Seqfish data.
        """

        """Read seqFISH expression matrix as dataframe."""
        df = pd.read_csv(path, dtype=np.uint16)

        X = csr_matrix(df)
        obs = pd.DataFrame(index=df.index.to_list())
        var = pd.DataFrame(index=df.columns.to_list())
    
        dtype = {
            "Field of View": np.uint8,
            "Cell ID": np.uint16,
            "X": np.float32,
            "Y": np.float32,
        "Region": "category",
        }

        """Read seqFISH cell centroid locations CSV as dataframe."""
        df_loc = pd.read_csv(
            meta_path,
            dtype=dtype,
        )

        rename = {
            "Field of View": "fov",
            "Cell ID": "cell_id",
            "X": "x",
            "Y": "y",
            "Region": "region",
        }
        df_loc = df_loc.rename(columns=rename)

        """Read fov_offset CSV as dataframe"""
        if fov_path is not None :
            fov_offset = pd.read_csv(fov_path) 
            if accumulate_x:
                for i in range(1, fov_offset.shape[0]):
                    fov_offset["x_offset"][i] = fov_offset["x_offset"][i] + fov_offset["x_offset"][i - 1]
            if accumulate_y:
                for i in range(1, fov_offset.shape[0]):
                    fov_offset["y_offset"][i] = fov_offset["y_offset"][i] + fov_offset["y_offset"][i - 1]

            for i in range(fov_offset.shape[0]):
                df_loc["x"][df_loc["fov"] == fov_offset["fov"][i]] = (
                df_loc["x"][df_loc["fov"] == fov_offset["fov"][i]] + fov_offset["x_offset"][i]
            )
                df_loc["y"][df_loc["fov"] == fov_offset["fov"][i]] = (
                df_loc["y"][df_loc["fov"] == fov_offset["fov"][i]] + fov_offset["y_offset"][i]
            )

        df_loc["spatial"] = [[int(df_loc["x"][i]), int(df_loc["y"][i])] for i in range(df_loc.shape[0])]

        """construct AnnData object."""
        lm.main_info("Constructing count matrix.")
        adata = AnnData(X=X, obs=obs, var=var)
        adata.obs["fov"] = df_loc["fov"].to_list()
        adata.obs["cell_id"] = df_loc["cell_id"].to_list()
        adata.obs["region"] = df_loc["region"].to_list()

        adata.obsm = pd.DataFrame(index=df_loc.index.to_list())
        adata.obsm["spatial"] = np.array(df_loc["spatial"].to_list())

        scale, scale_unit = 1.0, None

        # Set uns
        SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
        SKM.init_uns_pp_namespace(adata)
        SKM.init_uns_spatial_namespace(adata)
        SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, scale)
        SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, scale_unit)

        return adata
    

    #dataset from https://singlecell.broadinstitute.org/single_cell/study/SCP815/highly-sensitive-spatial-transcriptomics-at-near-cellular-resolution-with-slide-seqv2#study-download
    # ____  _     ___ ____  _____ ____  _____ ___  
    #/ ___|| |   |_ _|  _ \| ____/ ___|| ____/ _ \ 
    #\___ \| |    | || | | |  _| \___ \|  _|| | | |
    # ___) | |___ | || |_| | |___ ___) | |__| |_| |
    #|____/|_____|___|____/|_____|____/|_____\__\_\
                                               
    def slideseq(
        self,
        path: str, 
        beads_path: str, 
        binsize: Optional[int] = None, 
        version: Literal["slide2"] = "slide2"
    ) -> AnnData:
        """Read Slide-seq data as AnnData.

        Parameters:
        ----------------------------------------------------------------------
        path: Path to Slide-seq digital expression matrix CSV.
        beads_path: Path to CSV file containing bead locations.
        binsize: Size of pixel bins.
        version: Slideseq technology version. Currently only used to set the scale and
            scale units of each unit coordinate. This may change in the future.
            
        Returns:
        -----------------------------------------------------------------------
        AnnData object containing the slideseq data.
        """

        """Read a Slide-seq digital expression matrix."""
        data = pd.read_csv(path, sep="\t").rename(columns={"Row": "gene"})
        data = data.melt(id_vars="gene", var_name="barcode", value_name="count")
        data = data[data["count"] > 0]
        data["gene"] = data["gene"].astype("category")
        data["barcode"] = data["barcode"].astype("category")
        data["count"] = data["count"].astype(np.uint16)

        """Read a Slide-seq bead locations CSV file."""
        skiprows = None
        with ngs.utils.open_as_text(path, "r") as f:
            line = f.readline()
            if line.startswith("barcode"):
                skiprows = 1

        beads = pd.read_csv(beads_path, skiprows=skiprows, dtype={"barcode": "category"})
        beads = beads.rename(columns={"barcodes": "barcode", "xcoord": "x", "ycoord": "y"})
        data = pd.merge(data, beads, on="barcode")

        if binsize is not None:
            lm.main_info(f"Using binsize={binsize}")
            x_bin = bin_indices(data["x"].values, 0, binsize)
            y_bin = bin_indices(data["y"].values, 0, binsize)
            data["x"], data["y"] = x_bin, y_bin

            data["label"] = data["x"].astype(str) + "-" + data["y"].astype(str)
            props = get_bin_props(data[["x", "y", "label"]].drop_duplicates(), binsize)
        else:
            data.rename(columns = {"barcode": "label"}, inplace=True)
            props = (
                data[["x", "y", "label"]]
                .drop_duplicates()
                .set_index("label")
                .rename(columns = {"x": "centroid-0", "y": "centroid-1"})
            )

        uniq_gene = sorted(data["gene"].unique())
        uniq_cell = sorted(data["label"].unique())
        shape = (len(uniq_cell), len(uniq_gene))
        cell_dict = dict(zip(uniq_cell, range(len(uniq_cell))))
        gene_dict = dict(zip(uniq_gene, range(len(uniq_gene))))

        x_ind = data["label"].map(cell_dict).astype(int).values
        y_ind = data["gene"].map(gene_dict).astype(int).values

        lm.main_info("Constructing count matrix.")
        X = csr_matrix((data["count"].values, (x_ind, y_ind)), shape=shape)
        obs = pd.DataFrame(index=uniq_cell)
        var = pd.DataFrame(index=uniq_gene)
        adata = AnnData(X=X, obs=obs, var=var)
        ordered_props = props.loc[adata.obs_names]
        adata.obsm["spatial"] = ordered_props.filter(regex="centroid-").values

        scale, scale_unit = 1.0, None
        if version in VERSIONS:
            resolution = VERSIONS[version]
            scale, scale_unit = resolution.scale, resolution.unit

        # Set uns
        SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
        SKM.init_uns_pp_namespace(adata)
        SKM.init_uns_spatial_namespace(adata)
        SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_BINSIZE_KEY, binsize)
        SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, scale)
        SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, scale_unit)
    
        return adata


    #dataset from https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Human_Lymph_Node
    # __     _____ ____ ___ _   _ __  __ 
    # \ \   / /_ _/ ___|_ _| | | |  \/  |
    #  \ \ / / | |\___ \| || | | | |\/| |
    #   \ V /  | | ___) | || |_| | |  | |
    #    \_/  |___|____/___|\___/|_|  |_|
    def visium(
        self,
        matrix_dir: str, 
        meta_path: str, 
        version: Literal["visium"] = "visium"
    ) -> AnnData:
        """Read 10x Visium data as AnnData.

        Parameters:
        ----------------------------------------------------------------------------
            matrix_dir: Directory containing matrix files
            (barcodes.tsv.gz, features.tsv.gz, matrix.mtx.gz)
            meta_path: Path to CSV containing spatial coordinates
            version: 10x technology version. Currently only used to set the scale and
            scale units of each unit coordinate. This may change in the future.
        Returns:
        -----------------------------------------------------------------------------
        AnnData object containing the 10x visium data.
        """

        """Read 10x Visium matrix directory as AnnData. """
        obs = pd.read_csv(os.path.join(matrix_dir, "barcodes.tsv.gz"), names=["barcode"]).set_index("barcode")
        var = pd.read_csv(os.path.join(matrix_dir, "features.tsv.gz"), names=["gene_name", "gene_id", "library"]).set_index(
            "gene_id"
        )
        X = scipy.io.mmread(os.path.join(matrix_dir, "matrix.mtx.gz")).tocsr()
      
        """transpose gene expression matrix to create anndata object. """
        X = X.T
        adata = AnnData(X=X, obs=obs, var=var)

        """Read 10x tissue positions CSV as dataframe.
        https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/images
        """
        positions = pd.read_csv(
            meta_path, names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
        )
        adata.obs = positions.set_index("barcode").loc[adata.obs_names]
        adata.obsm["spatial"] = adata.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values

        scale, scale_unit = 1.0, None
        if version in VERSIONS:
            resolution = VERSIONS[version]
            scale, scale_unit = resolution.scale, resolution.unit

        # Set uns
        SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
        SKM.init_uns_pp_namespace(adata)
        SKM.init_uns_spatial_namespace(adata)
        # SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_BINSIZE_KEY, binsize)
        SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, scale)
        SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, scale_unit)
    
        return adata                      
    
    #dataset from
    # ____ _____ _____ ____  _____ ___  
    #/ ___|_   _| ____|  _ \| ____/ _ \ 
    #\___ \ | | |  _| | |_) |  _|| | | |
    # ___) || | | |___|  _ <| |__| |_| |
    #|____/ |_| |_____|_| \_\_____\___/ 
    def read_bgi_as_dataframe(
            self, 
            path: str, 
            label_column: Optional[str] = None
        ) -> pd.DataFrame:

        """Read a BGI read file as a pandas DataFrame.

        Args:
            path: Path to read file.
            label_column: Column name containing positive cell labels.

        Returns:
            Pandas Dataframe with the following standardized column names.
                * `gene`: Gene name/ID (whatever was used in the original file)
                * `x`, `y`: X and Y coordinates
                * `total`, `spliced`, `unspliced`: Counts for each RNA species.
                The latter two is only present if they are in the original file.
        """
        dtype = {
            "geneID": "category",  # geneID
            "x": np.uint32,  # x
            "y": np.uint32,  # y
            # Multiple different names for total counts
            "MIDCounts": np.uint16,
            "MIDCount": np.uint16,
            "UMICount": np.uint16,
            "UMICounts": np.uint16,
            "EXONIC": np.uint16,  # spliced
            "INTRONIC": np.uint16,  # unspliced,
        }
        rename = {
            "MIDCounts": "total",
            "MIDCount": "total",
            "UMICount": "total",
            "UMICounts": "total",
            "EXONIC": "spliced",
            "INTRONIC": "unspliced",
        }

        # Use first 10 rows for validation.
        df = pd.read_csv(path, sep="\t", dtype=dtype, comment="#", nrows=10)

        if label_column:
            dtype.update({label_column: np.uint32})
            rename.update({label_column: "label"})

            if label_column not in df.columns:
                raise IOError(f"Column `{label_column}` is not present.")

        # If duplicate columns are provided, we don't know which to use!
        rename_inverse = {}
        for _from, _to in rename.items():
            rename_inverse.setdefault(_to, []).append(_from)
        for _to, _cols in rename_inverse.items():
            if sum(_from in df.columns for _from in _cols) > 1:
                raise IOError(f"Found multiple columns mapping to `{_to}`.")

        return pd.read_csv(
            path,
            sep="\t",
            dtype=dtype,
            comment="#",
        ).rename(columns=rename)
    

    def dataframe_to_labels(self, df: pd.DataFrame, column: str, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Convert a BGI dataframe that contains cell labels to a labels matrix.

        Args:
            df: Read dataframe, as returned by :func:`read_bgi_as_dataframe`.
            columns: Column that contains cell labels as positive integers. Any labels
             that are non-positive are ignored.

        Returns:
            Labels matrix
        """
        shape = shape or (df["x"].max() + 1, df["y"].max() + 1)
        labels = np.zeros(shape, dtype=int)

        for label, _df in df.drop_duplicates(subset=[column, "x", "y"]).groupby(column):
            if label <= 0:
                continue
            labels[(_df["x"].values, _df["y"].values)] = label
        return labels
    

    def dataframe_to_filled_labels(self, df: pd.DataFrame, column: str, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Convert a BGI dataframe that contains cell labels to a (filled) labels matrix.

        Args:
            df: Read dataframe, as returned by :func:`read_bgi_as_dataframe`.
            columns: Column that contains cell labels as positive integers. Any labels
                that are non-positive are ignored.

        Returns:
            Labels matrix
        """
        shape = shape or (df["x"].max() + 1, df["y"].max() + 1)
        labels = np.zeros(shape, dtype=int)
        for label, _df in df.drop_duplicates(subset=[column, "x", "y"]).groupby(column):
            if label <= 0:
                continue
            points = _df[["x", "y"]].values.astype(int)
            min_offset = points.min(axis=0)
            max_offset = points.max(axis=0)
            xmin, ymin = min_offset
            xmax, ymax = max_offset
            points -= min_offset
            hull = cv2.convexHull(points, returnPoints=True)
            mask = cv2.fillConvexPoly(np.zeros((max_offset - min_offset + 1)[::-1], dtype=np.uint8), hull, color=1).T
            labels[xmin : xmax + 1, ymin : ymax + 1][mask == 1] = label
        return labels

    def bgi_agg(
        self,
        path: str,
        stain_path: Optional[str] = None,
        binsize: int = 1,
        gene_agg: Optional[Dict[str, Union[List[str], Callable[[str], bool]]]] = None,
        prealigned: bool = False,
        label_column: Optional[str] = None,
        version: Literal["stereo"] = "stereo",
    ) -> AnnData:
        """Read BGI read file to calculate total number of UMIs observed per
        coordinate.

        Args:
            path: Path to read file.
            stain_path: Path to nuclei staining image. Must have the same coordinate
                system as the read file.
            binsize: Size of pixel bins.
            gene_agg: Dictionary of layer keys to gene names to aggregate. For
                example, `{'mito': ['list', 'of', 'mitochondrial', 'genes']}` will
                yield an AnnData with a layer named "mito" with the aggregate total
                UMIs of the provided gene list.
            prealigned: Whether the stain image is already aligned with the minimum
                x and y RNA coordinates.
            label_column: Column that contains already-segmented cell labels.
            version: BGI technology version. Currently only used to set the scale and
                scale units of each unit coordinate. This may change in the future.

        Returns:
            An AnnData object containing the UMIs per coordinate and the nucleus
            staining image, if provided. The total UMIs are stored as a sparse matrix in
            `.X`, and spliced and unspliced counts (if present) are stored in
            `.layers['spliced']` and `.layers['unspliced']` respectively.
            The nuclei image is stored as a Numpy array in `.layers['nuclei']`.
        """
        lm.main_debug(f"Reading data from {path}.")
        data = self.read_bgi_as_dataframe(path, label_column)
        x_min, y_min = data["x"].min(), data["y"].min()
        x, y = data["x"].values, data["y"].values
        x_max, y_max = x.max(), y.max()
        shape = (x_max + 1, y_max + 1)

        # Read image and update x,y max if appropriate
        layers = {}
        if stain_path:
            lm.main_debug(f"Reading stain image from {stain_path}.")
            image = skimage.io.imread(stain_path)
            if prealigned:
                lm.main_warning(
                    (
                        "Assuming stain image was already aligned with the minimum x and y RNA coordinates. "
                        "(prealinged=True)"
                    )
                )
                image = np.pad(image, ((x_min, 0), (y_min, 0)))
            x_max = max(x_max, image.shape[0] - 1)
            y_max = max(y_max, image.shape[1] - 1)
            shape = (x_max + 1, y_max + 1)
            # Reshape image to match new x,y max
            if image.shape != shape:
                lm.main_warning(f"Padding stain image from {image.shape} to {shape} with zeros.")
                image = np.pad(image, ((0, shape[0] - image.shape[0]), (0, shape[1] - image.shape[1])))
            layers[SKM.STAIN_LAYER_KEY] = image

        # Construct labels matrix if present
        labels = None
        if "label" in data.columns:
            lm.main_warning("Using the `label_column` option may result in disconnected labels.")
            labels = self.dataframe_to_labels(data, "label", shape)
            layers[SKM.LABELS_LAYER_KEY] = labels

        if binsize > 1:
            lm.main_info(f"Binning counts with binsize={binsize}.")
            shape = (math.ceil(shape[0] / binsize), math.ceil(shape[1] / binsize))
            x = bin_indices(x, 0, binsize)
            y = bin_indices(y, 0, binsize)
            x_min, y_min = x.min(), y.min()

            # Resize image if necessary
            if stain_path:
                layers[SKM.STAIN_LAYER_KEY] = cv2.resize(image, shape[::-1])

            if labels is not None:
                lm.main_warning("Cell labels were provided, but `binsize` > 1. There may be slight inconsistencies.")
                layers[SKM.LABELS_LAYER_KEY] = labels[::binsize, ::binsize]

        # See read_bgi_as_dataframe for standardized column names
        lm.main_info("Constructing count matrices.")
        X = csr_matrix((data["total"].values, (x, y)), shape=shape, dtype=np.uint16)
        if "spliced" in data.columns:
            layers[SKM.SPLICED_LAYER_KEY] = csr_matrix((data["spliced"].values, (x, y)), shape=shape, dtype=np.uint16)
        if "unspliced" in data.columns:
            layers[SKM.UNSPLICED_LAYER_KEY] = csr_matrix((data["unspliced"].values, (x, y)), shape=shape, dtype=np.uint16)

        # Aggregate gene lists
        if gene_agg:
            lm.main_info("Aggregating counts for genes provided by `gene_agg`.")
            for name, genes in gene_agg.items():
                mask = data["geneID"].isin(genes) if isinstance(genes, list) else data["geneID"].map(genes)
                data_genes = data[mask]
                _x, _y = data_genes["x"].values, data_genes["y"].values
                layers[name] = csr_matrix(
                    (data_genes["total"].values, (_x, _y)),
                    shape=shape,
                    dtype=np.uint16,
                )

        adata = AnnData(X=X, layers=layers)[x_min:, y_min:].copy()

        scale, scale_unit = 1.0, None
        if version in VERSIONS:
            resolution = VERSIONS[version]
            scale, scale_unit = resolution.scale, resolution.unit

        # Set uns
        SKM.init_adata_type(adata, SKM.ADATA_AGG_TYPE)
        SKM.init_uns_pp_namespace(adata)
        SKM.init_uns_spatial_namespace(adata)
        SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_BINSIZE_KEY, binsize)
        SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, scale)
        SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, scale_unit)
        return adata
    
    @SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE, "segmentation_adata", optional=True)
    def bgi(
        self,
        path: str,
        binsize: Optional[int] = None,
        segmentation_adata: Optional[AnnData] = None,
        labels_layer: Optional[str] = None,
        labels: Optional[Union[np.ndarray, str]] = None,
        seg_binsize: int = 1,
        label_column: Optional[str] = None,
        add_props: bool = True,
        version: Literal["stereo"] = "stereo",
    ) -> AnnData:
        """Read BGI read file as AnnData.

        Args:
            path: Path to read file.
            binsize: Size of pixel bins. Should only be provided when labels
                (i.e. the `segmentation_adata` and `labels` arguments) are not used.
            segmentation_adata: AnnData containing segmentation results.
            labels_layer: Layer name in `segmentation_adata` containing labels.
            labels: Numpy array or path to numpy array saved with `np.save` that
                contains labels.
            seg_binsize: the bin size used in cell segmentation, used in conjunction
                with `labels` and will be overwritten when `labels_layer` and
                `segmentation_adata` are not None.
            label_column: Column that contains already-segmented cell labels. If this
                column is present, this takes prescedence.
            add_props: Whether or not to compute label properties, such as area,
                bounding box, centroid, etc.
            version: BGI technology version. Currently only used to set the scale and
                scale units of each unit coordinate. This may change in the future.

        Returns:
            Bins x genes or labels x genes AnnData.
        """
        # Check inputs
        if sum([binsize is not None, segmentation_adata is not None, labels is not None, label_column is not None]) != 1:
            raise IOError("Exactly one of `segmentation_adata`, `binsize`, `labels`, `label_column` must be provided.")
        if (segmentation_adata is None) ^ (labels_layer is None):
            raise IOError("Both `segmentation_adata` and `labels_layer` must be provided.")
        if segmentation_adata is not None:
            if SKM.get_adata_type(segmentation_adata) != SKM.ADATA_AGG_TYPE:
                raise IOError("Only `AGG` type AnnDatas are supported.")
        if binsize is not None and abs(int(binsize)) != binsize:
            raise IOError("Positive integer `binsize` must be provided when `segmentation_adata` is not provided.")
        if isinstance(labels, str):
            labels = np.load(labels)

        lm.main_debug(f"Reading data from {path}.")
        data = self.read_bgi_as_dataframe(path, label_column)
        n_columns = data.shape[1]

        # Obtain total genes from raw data, so that the columns always match
        # regardless of what method was used.
        uniq_gene = sorted(data["geneID"].unique())

        props = None
        if label_column is not None:
            lm.main_info(f"Using cell labels from `{label_column}` column.")
            binsize = 1
            data = data[data["label"] > 0]
            if add_props:
                lm.main_warning(
                    "Using `label_column` as cell labels with `add_props=True` may result in incorrect contours."
                )
                props = get_points_props(data[["x", "y", "label"]])

        elif binsize is not None:
            lm.main_info(f"Using binsize={binsize}")
            if binsize < 2:
                lm.main_warning("Please consider using a larger bin size.")

            if binsize > 1:
                x_bin = bin_indices(data["x"].values, 0, binsize)
                y_bin = bin_indices(data["y"].values, 0, binsize)
                data["x"], data["y"] = x_bin, y_bin

            data["label"] = data["x"].astype(str) + "-" + data["y"].astype(str)
            if add_props:
                props = get_bin_props(data[["x", "y", "label"]].drop_duplicates(), binsize)

        # Use labels.
        else:
            binsize = 1
            shape = (data["x"].max(), data["y"].max())
            if labels is not None:
                lm.main_info(f"Using labels provided with `labels` argument.")
                if labels.shape != shape:
                    lm.main_warning(f"Labels matrix {labels.shape} has different shape as data matrix {shape}")
            else:
                labels = SKM.select_layer_data(segmentation_adata, labels_layer)
            label_coords = get_coords_labels(labels)

            if labels_layer is not None:
                lm.main_info(f"Using labels provided with `segmentation_adata` and `labels_layer` arguments.")
                seg_binsize = SKM.get_uns_spatial_attribute(segmentation_adata, SKM.UNS_SPATIAL_BINSIZE_KEY)
                x_min, y_min = (
                    int(segmentation_adata.obs_names[0]) * seg_binsize,
                    int(segmentation_adata.var_names[0]) * seg_binsize,
                )
                label_coords["x"] += x_min
                label_coords["y"] += y_min

            # When binning was used for segmentation, need to expand indices to cover
            # every binned pixel.
            if seg_binsize > 1:
                lm.main_warning("Binning was used for segmentation.")
                coords_dfs = []
                for i in range(seg_binsize):
                    for j in range(seg_binsize):
                        coords = label_coords.copy()
                        coords["x"] += i
                        coords["y"] += j
                        coords_dfs.append(coords)
                label_coords = pd.concat(coords_dfs, ignore_index=True)
            data = pd.merge(data, label_coords, on=["x", "y"], how="inner")
            if add_props:
                props = get_label_props(labels)

        uniq_cell = sorted(data["label"].unique())
        shape = (len(uniq_cell), len(uniq_gene))
        cell_dict = dict(zip(uniq_cell, range(len(uniq_cell))))
        gene_dict = dict(zip(uniq_gene, range(len(uniq_gene))))
        x_ind = data["label"].map(cell_dict).astype(int).values
        y_ind = data["geneID"].map(gene_dict).astype(int).values

        # See read_bgi_as_dataframe for standardized column names
        lm.main_info("Constructing count matrices.")
        X = csr_matrix((data["total"].values, (x_ind, y_ind)), shape=shape)
        layers = {}
        if "spliced" in data.columns:
            layers[SKM.SPLICED_LAYER_KEY] = csr_matrix((data["spliced"].values, (x_ind, y_ind)), shape=shape)
        if "unspliced" in data.columns:
            layers[SKM.UNSPLICED_LAYER_KEY] = csr_matrix((data["unspliced"].values, (x_ind, y_ind)), shape=shape)

        obs = pd.DataFrame(index=uniq_cell)
        var = pd.DataFrame(index=uniq_gene)
        adata = AnnData(X=X, obs=obs, var=var, layers=layers)
        if props is not None:
            ordered_props = props.loc[adata.obs_names]
            adata.obs["area"] = ordered_props["area"].values
            adata.obsm["spatial"] = ordered_props.filter(regex="centroid-").values
            adata.obsm["contour"] = ordered_props["contour"].values
            adata.obsm["bbox"] = ordered_props.filter(regex="bbox-").values

        scale, scale_unit = 1.0, None
        if version in VERSIONS:
            resolution = VERSIONS[version]
            scale, scale_unit = resolution.scale, resolution.unit

        # Set uns
        SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
        SKM.init_uns_pp_namespace(adata)
        SKM.init_uns_spatial_namespace(adata)
        SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_BINSIZE_KEY, binsize)
        SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, scale)
        SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, scale_unit)
        return adata

    #dataset from https://singlecell.broadinstitute.org/single_cell/study/SCP1830/spatial-atlas-of-molecular-cell-types-and-aav-accessibility-across-the-whole-mouse-brain?utm_source=chatgpt.com#study-download
    #____ _____  _    ____  __  __    _    ____  
   #/ ___|_   _|/ \  |  _ \|  \/  |  / \  |  _ \ 
   #\___ \ | | / _ \ | |_) | |\/| | / _ \ | |_) |
   # ___) || |/ ___ \|  _ <| |  | |/ ___ \|  __/ 
   #|____/ |_/_/   \_\_| \_\_|  |_/_/   \_\_|   
    
    def starmap_plus(
        self,
        folder_path:str
    ) -> AnnData:
        """ Read StarMAP Plus output files and construct an AnnData object.
   
        Parameters:
        --------------------------------------------------------------------------
            folder_path:str
                Path to the folder containing STARmap Plus data files.
                The folder must include three files whose filenames contain the following keywords:
                - "spatial" --------> CSV file with spatial coordinates (e.g., well_2_5_spatial.csv)
                - "spot_meta"-------> CSV file with spot or cell metadata (e.g., well_2_5_spot_meta.csv)
                - "raw_expression_pd"------> CSV file with gene expression matrix (e.g., well_2_5processed_expression_pd.csv)
                
        Returns:
        ----------------------------------------------------------------------------
            AnnData object containing the STARmap Plus data.
            
        Raises:
        -----------------------------------------------------------------------------
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

    
    #dataset from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7990097"""
    # ___  ____  _____ _   _ ____ _____ 
   # / _ \|  _ \| ____| \ | / ___|_   _|
   #| | | | |_) |  _| |  \| \___ \ | |  
   #| |_| |  __/| |___| |\  |___) || |  
   # \___/|_|   |_____|_| \_|____/ |_|  
                                     
    def openST(
        self,
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

    def read_merfish(
            self, 
            path: str,
            meta_path: str,
        ) -> AnnData:
        return self.merfish(path,meta_path)

    def read_seqfish(
            self, 
            path: str,
            meta_path: str,
            fov_path: str | None = None,
            accumulate_x: bool = False,
            accumulate_y: bool = False,
        ) -> AnnData:
        return self.seqfish(path, meta_path, fov_path, accumulate_x, accumulate_y)
    
    def read_slideseq(
            self, 
            path: str, 
            beads_path: str, 
            binsize: Optional[int] = None, 
            version: Literal["slide2"] = "slide2"
        ) -> AnnData:
        return self.slideseq(path, beads_path, binsize, version)

    def read_bgi_agg(
            self, 
            path: str,
            stain_path: Optional[str] = None,
            binsize: int = 1,
            gene_agg: Optional[Dict[str, Union[List[str], Callable[[str], bool]]]] = None,
            prealigned: bool = False,
            label_column: Optional[str] = None,
            version: Literal["stereo"] = "stereo",
        ) -> AnnData:
        return self.bgi_agg(path, stain_path, binsize, gene_agg, prealigned, label_column, version)
    
    def read_bgi(
            self, 
            path: str,
            binsize: Optional[int] = None,
            segmentation_adata: Optional[AnnData] = None,
            labels_layer: Optional[str] = None,
            labels: Optional[Union[np.ndarray, str]] = None,
            seg_binsize: int = 1,
            label_column: Optional[str] = None,
            add_props: bool = True,
            version: Literal["stereo"] = "stereo",
        ) -> AnnData:
        return self.bgi(path, binsize, segmentation_adata, labels_layer, labels, seg_binsize, label_column, add_props, version)
    
    def read_visium(
            self, 
            matrix_dir: str, 
            meta_path: str, 
            version: Literal["visium"] = "visium"
            ) -> AnnData:
        return self.visium(matrix_dir, meta_path, version)
    
    def read_starmap(
            self, 
            folder_path: str
            ) -> AnnData:
        return self.starmap_plus(folder_path)
    
    def read_openST(
            self, 
            path: str
            ) -> AnnData:
        return self.openST(path)
    
    

