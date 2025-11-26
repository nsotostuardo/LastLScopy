from ..core.pipeline import Pipeline, RMSPipeline
import numpy as np
from scipy.ndimage import gaussian_filter
from numpy.typing import NDArray

class SciPyPipeline(Pipeline):

    #Hacer antes el espacial de para cada S espacial y despues los S espectrales.
    def gaussian_filtering(self, data:NDArray[np.float64], sigma:float, spatial_sigma:float ,mode:str="constant", cval:float=0.0, truncate:float=4.0):
        result = gaussian_filter(data, sigma= [sigma,spatial_sigma,spatial_sigma], mode=mode, cval= cval, truncate= truncate)
        return result
    

class SciPyRMS(RMSPipeline):

    def rms_filtering(self, data:NDArray[np.float64], use_mask= False) -> NDArray[np.float64]:
        print("Using default nanstd") 
        if use_mask:
            self.use_mask(data)

        for i in range(len(data)):
            data[i] = self.cp_rms(data,i)
               
        return(data)
               
    def cp_rms(self, data:NDArray[np.float64], index:int):
        initial_rms = np.nanstd(data[index])
        final_rms = np.nanstd(data[index][data[index]<5.0*initial_rms])
        return (data[index]/final_rms)