import typing

from core.ml.data.loaders.common.converter import Converter
from core.ml.data.loaders.common.dataloader import (
    Dataloader as VoxelDataloader,
)
from core.ml.data.loaders.common.registry import ConverterRegistry
from core.ml.data.loaders.resources.dataloaders.csv_dataloader import (
    ImageFromCSVDataloader,
)
from core.structs.dataset import Dataset as VoxelDataset


@ConverterRegistry.register()
class ImageCsvDatasetToPytorchImageDataloader(Converter):
    """
    Utility converter to convert between IMAGE_CSV formatted datasets
    and PYTORCH_IMAGE dataloaders
    """

    @classmethod
    def convert(
        cls, dataset: VoxelDataset, **kwargs: dict
    ) -> typing.Type[VoxelDataloader]:
        """
        Converts a dataset object into a pytorch image dataloader

        Args:
            dataset (VoxelDataset): the original dataset
            kwargs (dict): any downstream keyword arguments to be used in the converter
                           dataloader/dataset

        Returns:
            typing.type[VoxelDataloader]: the destination dataloader
        """
        dataset_dir = (
            dataset.get_download_path()
            if dataset.is_downloaded()
            else dataset.download()
        )
        return ImageFromCSVDataloader(directory=dataset_dir, **kwargs)
