import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from ..core.pipeline import Pipeline


class TorchPipeline(Pipeline):
    def __init__(self, args):
        super().__init__(args)
        self.device_torch = torch.device(args.TorchDevice) 
        self._kernel_cache: dict[tuple[float, float, torch.dtype, str], torch.Tensor] = {}

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
            raise ValueError("TorchPipeline only supports mode='constant'")
        if cval != 0.0:
            raise ValueError("TorchPipeline only supports cval=0.0")

        if sigma <= 0 and spatial_sigma <= 0:
            return self._normalize_numpy(data)

        with torch.no_grad():
            tensor = self._to_torch_tensor(data)

            if sigma > 0:
                kernel_z = self._get_1d_kernel(sigma, truncate, tensor.dtype)
                tensor = self._convolve_axis(tensor, kernel_z, axis=0)

            if spatial_sigma > 0:
                kernel_xy = self._get_1d_kernel(spatial_sigma, truncate, tensor.dtype)
                tensor = self._convolve_axis(tensor, kernel_xy, axis=1)
                tensor = self._convolve_axis(tensor, kernel_xy, axis=2)

            return tensor.squeeze(0).squeeze(0).cpu().numpy()

    def _normalize_numpy(self, data: NDArray[np.float64]) -> NDArray[np.float32]:
        data = np.asarray(data)
        if not data.dtype.isnative:
            data = data.byteswap().view(data.dtype.newbyteorder("="))
        return np.asarray(data, dtype=np.float32)

    def _to_torch_tensor(self, data: NDArray[np.float64]) -> torch.Tensor:
        native = self._normalize_numpy(data)
        if not native.flags.c_contiguous:
            native = np.ascontiguousarray(native)
        return torch.from_numpy(native).unsqueeze(0).unsqueeze(0).to(self.device_torch)

    def _get_1d_kernel(self, sigma: float, truncate: float, dtype: torch.dtype) -> torch.Tensor:
        key = (float(sigma), float(truncate), dtype, self.device_torch.type)
        kernel = self._kernel_cache.get(key)
        if kernel is None:
            kernel = gaussian_kernel_1d_torch(sigma, truncate=truncate, dtype=dtype)
            kernel = kernel.to(self.device_torch)
            self._kernel_cache[key] = kernel
        return kernel

    def _convolve_axis(self, data: torch.Tensor, kernel_1d: torch.Tensor, axis: int) -> torch.Tensor:
        radius = kernel_1d.shape[0] // 2
        if axis == 0:
            weight = kernel_1d.view(1, 1, -1, 1, 1)
            padding = (radius, 0, 0)
        elif axis == 1:
            weight = kernel_1d.view(1, 1, 1, -1, 1)
            padding = (0, radius, 0)
        elif axis == 2:
            weight = kernel_1d.view(1, 1, 1, 1, -1)
            padding = (0, 0, radius)
        else:
            raise ValueError(f"Invalid axis {axis} for 3D convolution")

        return F.conv3d(data, weight, padding=padding)


def gaussian_kernel_1d_torch(sigma: float, truncate: float = 4.0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if sigma <= 0:
        return torch.tensor([1.0], dtype=dtype)

    radius = int(truncate * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel
