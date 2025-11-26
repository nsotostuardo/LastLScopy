from ..core.pipeline import Pipeline
from numpy.typing import NDArray
from ..utils.CUDAdecorators import Vram_clean
import cupy as cp
import numpy as np

class CudaPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._module = None
        self._kernel_func = None
        self._compile_c()
    
    @Vram_clean
    def gaussian_filtering(self, data:NDArray[np.float64], sigma:float, truncate:float=4.0)-> NDArray[np.float64]:
        if sigma == 0:
            return data
        else: 
            data_gpu = cp.asarray(data)
            data_gpu = self.gaussian_filter_axis0_cupy(data_gpu, sigma, truncate)
            result = cp.asnumpy(data_gpu)
            return result
    
    def gaussian_filter_axis0_cupy(self, data, sigma, truncate=4.0):
        """
        Aplica un filtro gaussiano 1D en el eje espectral en GPU usando CUDA.

        Parámetros:
        - data: cp.ndarray (float32), cubo de datos en formato [ espectral, espacial, espacial]
        - sigma: desviación estándar de la Gaussiana en eje espectral, en espacial se asume nulo
        - truncate: define el tamaño del kernel como 2 * int(truncate * sigma) + 1

        Retorna:
        - np.ndarray con la convolución espectral
        """

        depth, height, width = data.shape
        kernel = self.gaussian_kernel_1d(sigma, truncate)
        ksize = kernel.shape[0] // 2

        output = cp.empty_like(data)

        block = (16, 16)
        grid = ((width + block[0] - 1) // block[0],
                (height + block[1] - 1) // block[1])

        self._kernel_func(grid, block,
            (data, output, kernel, cp.int32(depth), cp.int32(height), cp.int32(width), cp.int32(ksize))
        )

        return cp.asnumpy(output)
    
    def gaussian_kernel_1d(self, sigma, truncate=4.0):
        """Genera un kernel gaussiano 1D en GPU."""
        radius = int(truncate * sigma + 0.5)
        x = cp.arange(-radius, radius + 1, dtype=cp.float32)
        kernel = cp.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel /= cp.sum(kernel)
        return kernel


    def _compile_c(self) -> str:
        """
        Returns C code to be compiled with _compile_c
        """

        c_code = r'''
        extern "C" __global__
        void gaussian_convolve_axis0(const float* input, float* output, const float* kernel,
                                    int depth, int height, int width, int ksize)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            int z;

            if (x >= width || y >= height) return;

            for (z = 0; z < depth; ++z) {
                float sum = 0.0;
                for (int k = -ksize; k <= ksize; ++k) {
                    int zk = z + k;
                    if (zk >= 0 && zk < depth) {
                        int idx = zk * height * width + y * width + x;
                        sum += input[idx] * kernel[k + ksize];
                    }
                }
                int out_idx = z * height * width + y * width + x;
                output[out_idx] = sum;
            }
        }
        '''
        
        self._module = cp.RawModule(code=c_code)
        self._kernel_func = self._module.get_function('gaussian_convolve_axis0')
