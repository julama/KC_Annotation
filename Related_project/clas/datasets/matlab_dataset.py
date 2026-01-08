from kedro.io import AbstractDataset
from kedro.io.core import DatasetError
import scipy.io as sio


# === Binary Acc Dataset ==============================================
class MatlabDataset(AbstractDataset):
    """A custom dataset for loading matlab files.

    Allows to pass in additional arguments to the `scipy.io.loadmat` function.
    """
    def __init__(self, filepath: str, load_args: dict = {}):
        self._filepath = filepath
        self._load_args = load_args 

    def _load(self) -> bytes:
        # overwrite so that dataframe is returned
        return sio.loadmat(self._filepath, **self._load_args)

    def _save(self, data: bytes) -> None:
        raise DatasetError("Save operation is not implemented for MatlabDataset.")

    def _exists(self) -> bool:
        return os.path.exists(self._filepath)

    def _describe(self) -> dict:
        return dict(filepath=self._filepath)

