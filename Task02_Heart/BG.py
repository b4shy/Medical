import numpy as np
import matplotlib.pyplot as plt
import mriHandler

from batchgenerators.dataloading.data_loader import DataLoaderBase

class DataLoader(DataLoaderBase):
    def __init__(self, data, BATCH_SIZE=1, num_batches=None, seed=False):
        super(DataLoader, self).__init__(data, BATCH_SIZE, num_batches,seed)
        self.dh = mriHandler.MRIHandler()

    def generate_train_batch(self):

        return {'data': img, 'label': label, 'orig': orig}

    def load_data(self):


dh = mriHandler.MRIHandler()
data = dh.paths_to_all_imgs
batch_gen = DataLoader
