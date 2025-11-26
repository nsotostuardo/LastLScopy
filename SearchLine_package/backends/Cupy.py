from ..core.pipeline import Pipeline, RMSPipeline
from numpy.typing import NDArray
from cupyx.scipy.ndimage import gaussian_filter
from ..utils.CUDAdecorators import Vram_clean
import cupy as cp
import numpy as np  

class CuPyPipeline(Pipeline):
    #Hacer antes el Hacerlos Sigmas juntos
    @Vram_clean
    def gaussian_filtering(self, data:NDArray[np.float64], sigma:float, spatial_sigma:float, mode:str="constant", cval:float=0.0, truncate:float=4.0)-> NDArray[np.float64]:
        data_gpu = cp.asarray(data)
        data_gpu = gaussian_filter(data_gpu, sigma=[sigma, spatial_sigma, spatial_sigma], mode=mode, cval= cval, truncate= truncate)
        result = cp.asnumpy(data_gpu)
        return result
    
class CuPyRMS(RMSPipeline):

    @Vram_clean
    def rms_filtering(self, data:NDArray[np.float64], use_mask= False) -> NDArray[np.float64]:
        print("Using Cupy nanstd") 
        if use_mask:
            self.use_mask(data)

        data = cp.asarray(data)
        for i in range(len(data)):
            data[i] = self.cp_rms(data,i)
               
        return cp.asnumpy(data)
               
    def cp_rms(self, data:NDArray[np.float64], index:int):
        initial_rms = cp.nanstd(data[index])
        final_rms = cp.nanstd(data[index][data[index]<5.0*initial_rms])
        return (data[index]/final_rms)