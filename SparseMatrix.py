import logging
from pickle import dump, load

from numpy import array, ndarray
from numpy import where as np_where
from numpy.typing import ArrayLike
from pandas import DataFrame
from pathlib import Path
from scipy.sparse import csr_matrix, hstack, random, load_npz, save_npz, vstack
from tarfile import open as tar_open


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def create_tarfile(output_name: Path | str, source_dir: Path | str):
    """Creates a tar archive from a directory.

    Args:
        output_filename: The path to the output tar file.
        source_dir: The directory to archive.
    """

    output_name = Path(output_name)
    source_dir = Path(source_dir)
    with tar_open(output_name, "w:gz") as tar:
        tar.add(source_dir, arcname=source_dir.name)


def rmdir(path_: Path | str):
    """Removes a directory and tree files"""
    for root, dirs, files in path_.walk(top_down=False):
        for name in files:
            (root / name).unlink()
        for name in dirs:
            (root / name).rmdir()
    path_.rmdir()


class SparseDataFrame:
    def __init__(
        self,
        data: any = None,
        columns: ArrayLike | None = None,
        indices: ArrayLike | None = None,
        tar_file: str | Path = None,
    ) -> None:
        if tar_file:
            self.load(tar_file)
        else:
            self.__data = data
            self.__columns = columns
            self.__indices = indices
            self.update_dims()
        self.HIDDEN_PATH = Path(f".temp_sparse_files_/")

    def update_dims(self):
        self.n_cols = self.__data.shape[1]
        if self.__columns:
            ncols = len(self.__columns)
            assert (
                len(set(self.__columns)) == ncols
            ), "Error, columns names must be unique!"
            assert (
                self.n_cols == ncols
            ), "Error, number of columns must match first dimensionality of the data!"

        self.n_inds = self.__data.shape[0]
        if self.__indices:
            nrows = len(self.__indices)
            assert (
                len(set(self.__indices)) == nrows
            ), "Error, indices names must be unique!"
            assert (
                self.n_inds == nrows
            ), "Error, number of indices must match first dimensionality of the data!"

    def shape(self):
        return self.__data.shape

    def nrows(self):
        return self.n_inds

    def ncols(self):
        return self.n_cols

    def get_data(self, copy: bool = True):
        return self.__data.copy() if copy else self.__data

    def indices(self):
        return self.__indices

    def columns(self):
        return self.__columns

    def set_columns(self, columns: ArrayLike):
        assert isinstance(columns, (ndarray, set, list)), "Column type not recognized!"
        assert self.n_cols == len(
            columns
        ), "Error, number of columns must match zeroith dimensionality of the data!"
        if isinstance(columns, ndarray):
            self.__columns = columns.tolist()
        if isinstance(columns, set):
            self.__columns = list(columns)
        if isinstance(columns, list):
            self.__columns = columns
        self.update_dims()

    def set_indices(self, indices: ArrayLike):
        assert isinstance(indices, (ndarray, set, list)), "Column type not recognized!"
        assert self.n_inds == len(
            indices
        ), "Error, number of indices must match first dimensionality of the data!"
        if isinstance(indices, ndarray):
            self.__indices = indices.tolist()
        if isinstance(indices, set):
            self.__indices = list(indices)
        if isinstance(indices, list):
            self.__indices = indices
        self.update_dims()

    def __rm_icols(self, mask: list):
        mask = [x for x in range(self.n_cols) if x not in mask]
        self.__data = self.__data[mask, :]
        self.__columns = [x for i, x in enumerate(self.__columns) if i in mask]
        self.update_dims()

    def __rm_irows(self, mask: list):
        mask = [x for x in range(self.n_rows) if x not in mask]
        self.__data = self.__data[:, mask]
        self.__indices = [x for i, x in enumerate(self.__indices) if i in mask]
        self.update_dims()

    def rm_cols(self, columns: ArrayLike):
        assert self.__columns, "Columns not set!"
        mask = [self.__columns.index(x) for x in columns]
        self.__rm_icols(mask)

    def rm_icols(self, columns: ArrayLike):
        assert max(columns) < self.n_cols, "Removing index OOB!"
        self.__rm_icols(columns)

    def rm_rows(self, indices: ArrayLike):
        assert self.__indices, "Indices not set!"
        mask = [self.__indices.index(x) for x in indices]
        self.__rm_irows(mask)

    def rm_irows(self, columns: ArrayLike):
        assert max(columns) < self.n_cols, "Removing index OOB!"
        self.__rm_irows(columns)

    def append_cols(self, data: csr_matrix, column_names: list = None):
        if column_names:
            assert data.shape[1] == len(
                column_names
            ), "Data and columns are added at different lengths!"
            assert self.__columns, "Column names must exist before adding!"
            self.__columns.extend(column_names)
        self.__data = hstack([self.__data, data])
        self.update_dims()

    def append_inds(self, data: csr_matrix, index_names: list = None):
        if index_names:
            assert data.shape[0] == len(
                index_names
            ), "Data and indices are added at different lengths!"
            assert self.__indices, "Index names must exist before adding!"
            self.__indices.extend(index_names)
        self.__data = vstack([self.__data, data])
        self.update_dims()

    def head(self, n: int = 5):
        """Return the n upper left corner of the matrix"""
        return DataFrame(
            data=self.__data[:n, :n].todense(),
            index=self.__indices[:n] if self.__indices else None,
            columns=self.__columns[:n] if self.__columns else None,
        )

    def head_right(self, n: int = 5):
        """Return the n upper right corner of the matrix"""
        return DataFrame(
            data=self.__data[-n:, :n].todense(),
            index=self.__indices[:n] if self.__indices else None,
            columns=self.__columns[-n:] if self.__columns else None,
        )

    def tail(self, n: int = 5):
        """Return the n bottom left corner of the matrix"""
        return DataFrame(
            data=self.__data[:n, -n:].todense(),
            index=self.__indices[-n:] if self.__indices else None,
            columns=self.__columns[:n] if self.__columns else None,
        )

    def tail_right(self, n: int = 5):
        """Return the n bottom right corner of the matrix"""
        return DataFrame(
            data=self.__data[-n:, -n:].todense(),
            index=self.__indices[-n:] if self.__indices else None,
            columns=self.__columns[-n:] if self.__columns else None,
        )

    def save(self, save_path: Path | str, overwrite: bool = False):
        """Save sparse matrix to a tar file"""

        save_path = Path(save_path)
        tar_name = (
            save_path.parent / f"{save_path.name}.tar"
            if save_path.suffix != "tar"
            else save_path
        )
        assert (
            overwrite == True or not tar_name.exists()
        ), f"File {tar_name} already exists! Set overwrite to True to overwrite."

        self.HIDDEN_PATH.mkdir(parents=True, exist_ok=True)
        logging.info("Saving the data")
        save_npz((self.HIDDEN_PATH / "data.npz"), self.__data)

        logging.info("Saving the columns and indices")
        with open((self.HIDDEN_PATH / "columns.names"), "wb") as f:
            dump(self.__columns, f)
        with open((self.HIDDEN_PATH / "indices.names"), "wb") as f:
            dump(self.__indices, f)

        logging.info(f"Compressing to a tar file {tar_name}")
        create_tarfile(output_name=tar_name, source_dir=self.HIDDEN_PATH)
        rmdir(self.HIDDEN_PATH)
        logging.info("Save complete")

    def load(self, load_path: Path | str):
        """Load sparse matrix from a tar file"""
        load_path = Path(load_path)
        self.HIDDEN_PATH = Path(f".temp_sparse_files_/")
        tar = tar_open(load_path, "r:gz")

        logging.info("Extracting file")
        tar.extractall(".")

        logging.info("Loading the data")
        self.__data = load_npz(self.HIDDEN_PATH / "data.npz")
        logging.info("Loading the columns and indices")
        with open((self.HIDDEN_PATH / "columns.names"), "rb") as f:
            self.__columns = load(f)
        with open((self.HIDDEN_PATH / "indices.names"), "rb") as f:
            self.__indices = load(f)
        rmdir(self.HIDDEN_PATH)
        logging.info("Load complete")
