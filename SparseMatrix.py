import logging
from pathlib import Path
from pickle import load
from tarfile import open as tar_open

from numpy import array
from numpy.typing import ArrayLike
from pandas import DataFrame
from scipy.sparse import csr_matrix, hstack, vstack


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
            self.data = data
            assert (
                not columns or len(columns) == data.shape[1]
            ), "Dimension of columns must match data"
            assert (
                not indices or len(indices) == data.shape[0]
            ), "Dimension of indices must match data"
            self.columns = columns
            self.indices = indices
            self.__update()
        self.HIDDEN_PATH = Path(f".temp_sparse_files_/")

    def __update(self) -> None:
        self.shape = self.data.shape

    def get_columns(self, columns: ArrayLike) -> csr_matrix:
        """Return data by column names, if column names are set. Otherwise return data by numerical column position"""
        if self.columns:
            i_loc = [i for i, c in enumerate(self.columns) if c in columns]
            if len(i_loc) == 0:
                logging.warning("No positions correspond to provides columns.")
                logging.warning(
                    "Consider checking argument for typos or type differences."
                )
            return self.data[:, i_loc]
        return self.data[:, columns]

    def get_indices(self, indices: ArrayLike) -> csr_matrix:
        """Return data by index names, if index names are set. Otherwise return data by numerical index position"""
        if self.indices:
            i_loc = [i for i, c in enumerate(self.indices) if c in indices]
            if len(i_loc) == 0:
                logging.warning("No positions correspond to provided indices.")
                logging.warning(
                    "Consider checking argument for typos or type differences."
                )
            return self.data[i_loc, :]
        return self.data[indices, :]

    def pd_head(self, n_: int = 5) -> DataFrame:
        df = DataFrame(
            data=self.data[:n_, :].toarray(),
            columns=self.columns,
            index=self.indices[:n_],
        )
        return df

    def pd_tail(self, n_: int = 5) -> DataFrame:
        df = DataFrame(
            data=self.data[-n_:, :].toarray(),
            columns=self.columns,
            index=self.indices[-n_:],
        )
        return df

    def head(self, n_: int = 5) -> DataFrame:
        """Alias for pd_head"""
        return self.pd_head(n_=n_)

    def tail(self, n_: int = 5) -> DataFrame:
        """Alias for pd_tail"""
        return self.pd_tail(n_=n_)

    def sp_head(self, n_: int = 5) -> csr_matrix:
        return self.data[:n_, :]

    def sp_tail(self, n_: int = 5) -> csr_matrix:
        return self.data[-n_:, :]

    def drop_columns(self, columns: ArrayLike, inplace: bool = False) -> None:
        keep_columns = [x for x in self.columns if x not in columns]
        if inplace:
            self.data = self.get_columns(columns=keep_columns)
            self.columns = keep_columns
            self.__update()
            return None
        else:
            return SparseDataFrame(
                data=self.get_columns(columns=keep_columns),
                columns=keep_columns,
                indices=self.indices,
            )

    def drop_indices(self, indices: ArrayLike, inplace: bool = False) -> None:
        keep_indices = [x for x in self.indices if x not in indices]
        if inplace:
            self.data = self.get_indices(indices=keep_indices)
            self.indices = keep_indices
            self.__update()
            return None
        else:
            return SparseDataFrame(
                data=self.get_indices(indices=keep_indices),
                columns=self.columns,
                indices=keep_indices,
            )

    # TODO: Type hinting is not working for the SparseDataFrame, debug
    def hstack(self, right_matrix, append_suffix: str = "r"):
        """Column-wise stacking of SparseDataFrame"""
        assert (
            self.indices == right_matrix.indices
        ), "Indices of column-wise stacked objects must match"
        right_columns = [
            f"{x}_{append_suffix}" if x in self.columns else x
            for x in right_matrix.columns
        ]
        return SparseDataFrame(
            data=hstack([self.data, right_matrix.data]),
            columns=(self.columns + right_columns),
            indices=self.indices,
        )

    def vstack(self, right_matrix, append_suffix: str = "r"):
        """Row-wise stacking of SparseDataFrame"""
        assert (
            self.columns == right_matrix.columns
        ), "Indices of row-wise stacked objects must match"
        right_indices = [
            f"{x}_{append_suffix}" if x in self.indices else x
            for x in right_matrix.indices
        ]
        return SparseDataFrame(
            data=vstack([self.data, right_matrix.data]),
            columns=self.columns,
            indices=(self.indices + right_indices),
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
        save_npz((self.HIDDEN_PATH / "data.npz"), self._data)

        logging.info("Saving the columns and indices")
        with open((self.HIDDEN_PATH / "columns.names"), "wb") as f:
            dump(self._columns, f)
        with open((self.HIDDEN_PATH / "indices.names"), "wb") as f:
            dump(self._indices, f)

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
        self._data = load_npz(self.HIDDEN_PATH / "data.npz")
        logging.info("Loading the columns and indices")
        with open((self.HIDDEN_PATH / "columns.names"), "rb") as f:
            self._columns = load(f)
        with open((self.HIDDEN_PATH / "indices.names"), "rb") as f:
            self._indices = load(f)
        rmdir(self.HIDDEN_PATH)
        logging.info("Load complete")
