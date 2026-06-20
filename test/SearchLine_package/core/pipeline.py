from abc import ABC, abstractmethod
from ..core.functions import get_mask
from numpy.typing import NDArray
import numpy as np


class Pipeline(ABC):
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def gaussian_filtering(self, data:NDArray[np.float64], sigma:float, ss_sigma:float) -> NDArray[np.float64] :
        """
        Applies a Gaussian Filter to the data cube on spectral axis.
        
        Parameters:
        - data: data cube to be used, axis format [spectral, spatial, spatial].
        - sigma: Standard deviation for the Gaussian filter.

        Return:
        - np.ndarray with gaussian filter
        """
        pass
    

class RMSPipeline(ABC):
    def __init__(self, args):
        self.args = args

    def use_mask(self, data:NDArray[np.float64], mask_value = np.nan) -> NDArray[np.float64]:
        """
        Applies mask to data using np.nan values.
        
        Parameters:
        - data: data cube to be used, axis format [spectral, spatial, spatial].
        - mask: S.

        Return:
        - np.ndarray
        """
        mask = get_mask(self.args.ContinuumImage, self.args.MaskSN)
        data[:, mask] = mask_value

        return data

    @abstractmethod
    def rms_filtering(self, data:NDArray[np.float64], use_mask= False, mask = []) -> NDArray[np.float64]:
        """
        Applies a Gaussian Filter to the data cube on spectral axis.
        
        Parameters:
        - data: data cube to be used, axis format [spectral, spatial, spatial].
        - sigma: Standard deviation for the Gaussian filter.

        Return:
        - np.ndarray with gaussian filter
        """
        pass