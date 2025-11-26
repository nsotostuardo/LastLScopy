from ..core.pipeline import Pipeline, RMSPipeline
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from ..core.functions import get_mask

@njit(parallel=True)
def single_convolve(data:NDArray[np.float64], weights:NDArray[np.float64]) ->NDArray[np.float64]:
    """
    Convolves the datacube with Gaussian kernel parallelized using Numbda njit parallel (Threading) 

    Parameters:
    - data: data cube to be used, axis format [spatial, spatial, convolution axis]
    - weights: 
    """

    z, y, x = data.shape
    radius = len(weights) // 2
    output = np.zeros_like(data)

    for j in prange(y):
        for i in prange(x):
            for k in range(z):
                val = 0.0
                for offset in range(-radius, radius + 1):
                    zk = k - offset
                    if 0 <= zk < z:
                        val += weights[offset + radius] * data[zk, j, i]
                output[k, j, i] = val
    return output


class NumbaPipeline(Pipeline):
    def gaussian_filtering(self, data, sigma):
        if sigma == 0 :
            return data
        else:
            kernel = self._1Dgaussian_kernel(sigma)
            data = np.array(data, dtype='<f4')
            result = single_convolve(data, kernel)
            return result
    
    def _1Dgaussian_kernel(self, sigma:float, truncate:float=4.0) -> NDArray[np.float64]:
        """
        Generate a 1-D Gaussian kernel to be used as weight for convolution.

        Parameters:
        -sigma: Standard deviation for the Gaussian kernel
        -truncate: Truncate the filter at this many standard deviations

        Returns:
        - 1D np.array normalized kernel 
        """
        radius = int(truncate * sigma + 0.5)
        x = np.arange(-radius, radius + 1)
        w = np.exp(-0.5 * (x / sigma)**2)
        w /= w.sum()
        return w

    
@njit
def nanmean_std(arr):
    count = 0
    total = 0.0
    for val in arr:
        if not np.isnan(val):
            total += val
            count += 1
    if count == 0:
        return 0.0, 0.0
    mean = total / count

    sq_diff = 0.0
    for val in arr:
        if not np.isnan(val):
            sq_diff += (val - mean) ** 2
    std = (sq_diff / count) ** 0.5
    return mean, std

@njit
def RMS_process( slice_2d):
    flat = slice_2d.ravel()
    _, initial_std = nanmean_std(flat)
    threshold = np.float32(5.0) * np.float32(initial_std)

    count = 0
    total = 0.0
    for val in flat:
        if not np.isnan(val) and val < threshold:
            total += val
            count += 1
    if count == 0:
        return slice_2d.astype(np.float32)

    mean = total / count
    sq_diff = 0.0
    for val in flat:
        if not np.isnan(val) and val < threshold:
            sq_diff += (val - mean) ** 2
    final_std = (sq_diff / count) ** 0.5
    if final_std == 0.0:
        return slice_2d.astype(np.float32)

    return (slice_2d / np.float32(final_std)).astype(np.float32)

@njit(parallel=True)
def process_cube_nomask( data):
    for i in prange(data.shape[0]):
        data[i] = RMS_process(data[i])


@njit(parallel=True)
def process_cube_with_mask( data, mask):
    for i in prange(data.shape[0]):
        for j in range(mask.shape[0]):
            for k in range(mask.shape[1]):
                if mask[j, k]:
                    data[i, j, k] = np.nan
        data[i] = RMS_process(data[i])

class NumbaRMS(RMSPipeline):

    def use_mask(self, data:NDArray[np.float64], mask_value = np.nan) -> NDArray[np.float64]:
        mask = get_mask(self.args.ContinuumImage, self.args.MaskSN)
        data[:, mask] = mask_value
        return data

    def rms_filtering(self, data:NDArray[np.float64], use_mask= False) -> NDArray[np.float64]:
        print("Using Numba nanstd") 
        data = np.array(data, dtype='<f4')
        if use_mask:
            mask = self.use_mask(data)
            process_cube_with_mask(data, mask)

        else:
            process_cube_nomask(data)

        return(data)
    
    