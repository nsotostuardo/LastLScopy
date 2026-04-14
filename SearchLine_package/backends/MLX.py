import numpy as np
import mlx.core as mx
from numpy.typing import NDArray

from ..core.pipeline import Pipeline


class MLXPipeline(Pipeline):
    def __init__(self, args):
        super().__init__(args)
        self.device_mlx = self._getdevice(args.DeviceMLX)
        self._kernel_cache: dict[tuple[float, float, object], mx.array] = {}
        self.spatial_chunk_size = getattr(args, "MLXSpatialChunkSize", None)

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
            cubo = self._convolve_axis_spatially_chunked(cubo, kernel_z, axis=0)

        if spatial_sigma > 0:
            kernel_xy = self._get_1d_kernel(spatial_sigma, truncate, cubo.dtype)
            cubo = self._convolve_axis_spatially_chunked(cubo, kernel_xy, axis=1)
            cubo = self._convolve_axis_spatially_chunked(cubo, kernel_xy, axis=2)

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

    def _convolve_axis_spatially_chunked(self, data: mx.array, kernel_1d: mx.array, axis: int) -> mx.array:
        ny = int(data.shape[2])
        nx = int(data.shape[3])
        chunk_size = self._resolve_spatial_chunk_size(data)
        if chunk_size >= ny and chunk_size >= nx:
            return self._convolve_axis(data, kernel_1d, axis=axis)

        halo_y = int(kernel_1d.shape[0] // 2) if axis == 1 else 0
        halo_x = int(kernel_1d.shape[0] // 2) if axis == 2 else 0

        rows = []
        for y_start in range(0, ny, chunk_size):
            y_stop = min(y_start + chunk_size, ny)
            read_y_start = max(0, y_start - halo_y)
            read_y_stop = min(ny, y_stop + halo_y)

            cols = []
            for x_start in range(0, nx, chunk_size):
                x_stop = min(x_start + chunk_size, nx)
                read_x_start = max(0, x_start - halo_x)
                read_x_stop = min(nx, x_stop + halo_x)

                chunk = data[:, :, read_y_start:read_y_stop, read_x_start:read_x_stop, :]
                chunk = self._convolve_axis(chunk, kernel_1d, axis=axis)

                crop_y_start = y_start - read_y_start
                crop_y_stop = crop_y_start + (y_stop - y_start)
                crop_x_start = x_start - read_x_start
                crop_x_stop = crop_x_start + (x_stop - x_start)
                chunk = chunk[:, :, crop_y_start:crop_y_stop, crop_x_start:crop_x_stop, :]
                cols.append(chunk)

            row = mx.concatenate(cols, axis=3)
            mx.eval(row)
            rows.append(row)

        result = mx.concatenate(rows, axis=2)
        mx.eval(result)
        return result

    def _resolve_spatial_chunk_size(self, data: mx.array) -> int:
        if self.spatial_chunk_size is not None:
            return max(1, int(self.spatial_chunk_size))

        spectral_bytes = int(data.shape[1]) * 4
        if spectral_bytes <= 0:
            return max(int(data.shape[2]), int(data.shape[3]))

        # Estimate a square spatial tile whose raw tensor footprint stays moderate.
        target_bytes = 256 * 1024 * 1024
        approx_area = max(1, target_bytes // spectral_bytes)
        chunk_size = int(max(32, np.sqrt(approx_area)))
        return min(chunk_size, int(data.shape[2]), int(data.shape[3]))


def gaussian_kernel_1d_mlx(sigma: float, truncate: float = 4.0, dtype=mx.float32) -> mx.array:
    if sigma <= 0:
        return mx.array([1.0], dtype=dtype)

    radius = int(truncate * sigma + 0.5)
    x = mx.arange(-radius, radius + 1, dtype=dtype)
    kernel = mx.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel
