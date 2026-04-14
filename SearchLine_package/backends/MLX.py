import numpy as np
import mlx.core as mx
from numpy.typing import NDArray

from ..core.pipeline import Pipeline


class MLXPipeline(Pipeline):
    def __init__(self, args):
        super().__init__(args)
        self.device_mlx = self._getdevice(args.DeviceMLX)
        self._kernel_cache: dict[tuple[float, float, object], mx.array] = {}

    def _getdevice(self, mode):
        return mx.gpu if mode == "gpu" else mx.cpu

    def gaussian_filtering(
        self,
        data: NDArray[np.float64],
        sigma: float,
        spatial_sigma: float,
        mode: str = "constant",
        cval: float = 0.0,
        truncate: float = 4.0,
    ) -> NDArray[np.float32]:
        if mode != "constant":
            raise ValueError("MLXPipeline only supports mode='constant'")
        if cval != 0.0:
            raise ValueError("MLXPipeline only supports cval=0.0")

        if sigma <= 0 and spatial_sigma <= 0:
            return self._normalize_numpy(data)

        cubo = self._to_mlx_array(data)

        if sigma > 0:
            kernel_z = self._get_1d_kernel(sigma, truncate, cubo.dtype)
            cubo = self._convolve_axis(cubo, kernel_z, axis=0)

        if spatial_sigma > 0:
            kernel_xy = self._get_1d_kernel(spatial_sigma, truncate, cubo.dtype)
            cubo = self._convolve_axis(cubo, kernel_xy, axis=1)
            cubo = self._convolve_axis(cubo, kernel_xy, axis=2)

        mx.eval(cubo)
        res_np = np.array(cubo)
        return res_np[0, :, :, :, 0]

    def _normalize_numpy(self, data: NDArray[np.float64]) -> NDArray[np.float32]:
        data = np.asarray(data)
        if not data.dtype.isnative:
            data = data.byteswap().view(data.dtype.newbyteorder("="))
        return np.asarray(data, dtype=np.float32)

    def _to_mlx_array(self, data: NDArray[np.float64]) -> mx.array:
        native = self._normalize_numpy(data)
        cubo = mx.array(native)
        cubo = mx.expand_dims(cubo, axis=0)
        cubo = mx.expand_dims(cubo, axis=-1)
        return cubo

    def _get_1d_kernel(self, sigma: float, truncate: float, dtype) -> mx.array:
        key = (float(sigma), float(truncate), dtype)
        kernel = self._kernel_cache.get(key)
        if kernel is None:
            kernel = gaussian_kernel_1d_mlx(sigma, truncate=truncate, dtype=dtype)
            self._kernel_cache[key] = kernel
        return kernel

    def _convolve_axis(self, data: mx.array, kernel_1d: mx.array, axis: int) -> mx.array:
        radius = int(kernel_1d.shape[0] // 2)
        if axis == 0:
            weight = mx.reshape(kernel_1d, (1, kernel_1d.shape[0], 1, 1, 1))
            padding = (radius, 0, 0)
        elif axis == 1:
            weight = mx.reshape(kernel_1d, (1, 1, kernel_1d.shape[0], 1, 1))
            padding = (0, radius, 0)
        elif axis == 2:
            weight = mx.reshape(kernel_1d, (1, 1, 1, kernel_1d.shape[0], 1))
            padding = (0, 0, radius)
        else:
            raise ValueError(f"Invalid axis {axis} for 3D convolution")

        return mx.conv3d(data, weight, padding=padding, stream=self.device_mlx)


def gaussian_kernel_1d_mlx(sigma: float, truncate: float = 4.0, dtype=mx.float32) -> mx.array:
    if sigma <= 0:
        return mx.array([1.0], dtype=dtype)

    radius = int(truncate * sigma + 0.5)
    x = mx.arange(-radius, radius + 1, dtype=dtype)
    kernel = mx.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel
