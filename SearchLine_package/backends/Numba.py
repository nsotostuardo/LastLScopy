from ..core.pipeline import Pipeline, RMSPipeline
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from ..core.functions import get_mask

def convolve_Z(data, kernel_z):
    tmp = np.moveaxis(data, 0, 2)   
    tmp = convolve_last_axis(tmp, kernel_z)
    return np.moveaxis(tmp, 2, 0)   

def convolve_Y(data, kernel_xy):
    tmp = np.moveaxis(data, 1, 2)   
    tmp = convolve_last_axis(tmp, kernel_xy)
    return np.moveaxis(tmp, 2, 1)

def convolve_X(data, kernel_xy):
    tmp = np.moveaxis(data, 2, 0)   
    tmp = convolve_last_axis(tmp, kernel_xy)
    return np.moveaxis(tmp, 0, 2)   

@njit(parallel=True, fastmath=True)
def convolve_last_axis(data, weights):
    """
    Convolution over the LAST axis of data.
    data: shape (A, B, C)
    weights: 1D kernel
    """
    A, B, C = data.shape
    r = weights.size // 2
    out = np.zeros_like(data)

    for i in prange(A):
        for j in range(B):
            row = data[i, j]
            out_row = out[i, j]

            for k in range(C):
                val = 0.0
                k_min = max(0, k - r)
                k_max = min(C - 1, k + r)

                for zk in range(k_min, k_max + 1):
                    val += row[zk] * weights[zk - k + r]

                out_row[k] = val

    return out

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
        return slice_2d.astype(np.float32, copy=False)

    mean = total / count
    sq_diff = 0.0
    for val in flat:
        if not np.isnan(val) and val < threshold:
            sq_diff += (val - mean) ** 2
    final_std = (sq_diff / count) ** 0.5
    if final_std == 0.0:
        return slice_2d.astype(np.float32, copy=False)

    return (slice_2d / np.float32(final_std)).astype(np.float32, copy=False)

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


    
class NumbaPipeline(Pipeline):

    def gaussian_filtering(self, data:NDArray[np.float64], sigma:float, spatial_sigma:float,  truncate:float=4.0)-> NDArray[np.float64]:

        data = np.asarray(data, dtype=np.float32)

        kernel_z  = self._1Dgaussian_kernel(sigma, truncate).astype(np.float32, copy=False)
        kernel_xy = self._1Dgaussian_kernel(spatial_sigma, truncate).astype(np.float32, copy=False)

        if sigma > 0:
            data = convolve_Z(data, kernel_z)

        if spatial_sigma > 0:
            data = convolve_Y(data, kernel_xy)

        if spatial_sigma > 0:
            data = convolve_X(data, kernel_xy)

        return data
    
    def _1Dgaussian_kernel(self, sigma:float, truncate:float=4.0) -> np.ndarray: 
        """ 
        Generate a 1-D Gaussian kernel to be used as weight for convolution. 
        Parameters:
            -sigma: Standard deviation for the Gaussian kernel 
            -truncate: Truncate the filter at this many standard deviations Returns: 
            - 1D np.array normalized kernel 
        """ 
        if sigma <= 0:
            return np.array([1.0], dtype=np.float32)
        
        radius = int(truncate * sigma + 0.5) 
        x = np.arange(-radius, radius + 1, dtype=np.float32) 
        w = np.exp(-0.5 * (x / sigma)**2).astype(np.float32, copy=False)
        w /= w.sum() 
        return w
    
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