from ..core.pipeline import PipeLine
import numpy as np
from scipy.ndimage import gaussian_filter
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN


class SciPyPipeline(PipeLine):

    def clustering(self, eps, COORD, min_samples = 1, leaf_size= 20):
        db = DBSCAN(eps=eps, min_samples=min_samples,leaf_size=leaf_size).fit(COORD)
        return db