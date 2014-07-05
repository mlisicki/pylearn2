import logging
import warnings
import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.config import yaml_parse
from pylearn2.datasets import control

logger = logging.getLogger(__name__)

class PreprocessedDataset(DenseDesignMatrix):
    def __init__(self,
                 preprocessed_dataset,
                 convert_to_one_hot=True,
                 start=None,
                 stop=None,
                 axes=['b', 0, 1, 'c']):
        self.args = locals()

        self.preprocessed_dataset = preprocessed_dataset
        self.rng = self.preprocessed_dataset.rng
        self.data_specs = preprocessed_dataset.data_specs
        self.X_space = preprocessed_dataset.X_space
        self.X_topo_space = preprocessed_dataset.X_topo_space
        self.view_converter = preprocessed_dataset.view_converter

        self.y = preprocessed_dataset.y
        self.y_labels = preprocessed_dataset.y_labels
        if convert_to_one_hot:
            if not (self.y.min() == 0):
                raise AssertionError("Expected y.min == 0 but y.min == %g" %
                                     self.y.min())
            nclass = self.y.max() + 1
            y = np.zeros((self.y.shape[0], nclass), dtype='float32')
            for i in xrange(self.y.shape[0]):
                y[i, self.y[i]] = 1.
            self.y = y
            assert self.y is not None
            space, source = self.data_specs
            space.components[source.index('targets')].dim = nclass

        if control.get_load_data():
            if start is not None:
                self.X = preprocessed_dataset.X[start:stop, :]
                if self.y is not None:
                    self.y = self.y[start:stop, :]
                assert self.X.shape[0] == stop-start
            else:
                self.X = preprocessed_dataset.X
        else:
            self.X = None
        if self.X is not None:
            if self.y is not None:
                assert self.y.shape[0] == self.X.shape[0]

        self.view_converter.set_axes(axes)
