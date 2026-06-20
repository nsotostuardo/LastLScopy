from numpy.typing import NDArray
from ..utils.CUDAdecorators import Vram_clean
from ..core.pipeline import PipeLine
from cuml.cluster import DBSCAN as cuDBSCAN
from sklearn.cluster import DBSCAN
import cudf
import cupy as cp

class CuPyPipeline(PipeLine):
    @Vram_clean
    def clustering(self, eps, COORD, min_samples = 1, leaf_size= 20):
        COORD_gpu = cudf.DataFrame(cp.asarray(COORD))
        db = cuDBSCAN(eps=eps, min_samples=min_samples, leaf_size=leaf_size).fit(COORD_gpu)

        labels = db.labels_.to_numpy()
        core_sample_indices = db.core_sample_indices_.to_numpy()

        db_cpu = DBSCAN(eps=eps, min_samples=min_samples, leaf_size=leaf_size)
        setattr(db_cpu, "labels_", labels)
        setattr(db_cpu, "core_sample_indices_", core_sample_indices)
        return db_cpu
    