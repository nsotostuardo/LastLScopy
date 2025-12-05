import cupy as cp
def Vram_clean(func):
    """
    Cleans allocated vram 
    """
    def wrapper(*args, **kwargs):
        resultado = func(*args, **kwargs)

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        return resultado
    
    return wrapper

def compile_c() -> list:
    """
    Returns C code to be compiled with _compile_c
    """

    c_code = r'''
    extern "C" {

    
    //   1D convolution (axis 0)
    // ============================
    __global__ void gaussian_convolve_axis0(
        const float* input, float* output, const float* kernel,
        int depth, int height, int width, int ksize)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) return;

        for (int z = 0; z < depth; ++z) {
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

    
    //   2D convolution (axis 1&2)
    // ============================
    __global__ void gaussian_convolve_spatial_2d(
        const float* input, float* output, const float* kernel,
        int depth, int height, int width, int ksize)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z;

        if (x >= width || y >= height || z >= depth) return;

        float sum = 0.0f;

        // Gaussiana 2D separable: kernel2D = ky * kx
        for (int dy = -ksize; dy <= ksize; ++dy) {
            int yy = y + dy;
            if (yy < 0) yy = 0;
            if (yy >= height) yy = height - 1;
            float wy = kernel[dy + ksize];

            for (int dx = -ksize; dx <= ksize; ++dx) {
                int xx = x + dx;
                if (xx < 0) xx = 0;
                if (xx >= width) xx = width - 1;
                float wx = kernel[dx + ksize];

                int idx = z * (height * width) + yy * width + xx;
                sum += input[idx] * wy * wx;
            }
        }

        int out_idx = z * (height * width) + y * width + x;
        output[out_idx] = sum;
    }

    } // extern "C"
    '''
    
    _module = cp.RawModule(code=c_code)
    _kernel_func = _module.get_function('gaussian_convolve_axis0')
    _kernel_spatial = _module.get_function('gaussian_convolve_spatial_2d')
    return([_module, _kernel_func, _kernel_spatial])