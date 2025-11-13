import numpy as np
import matplotlib.pyplot as plt

class Patch():

    def __init__(self, crop, index, col_edges, row_edges):

        self.crop = crop
        self.index = index

        self.col_edges = np.array(col_edges)
        self.row_edges = np.array(row_edges)
        self.roi_col_edges = self.col_edges - crop.left
        self.roi_row_edges = self.row_edges - crop.start

        self.width = col_edges[1] - col_edges[0]
        self.height = row_edges[1] - row_edges[0]

        self.image = crop.signal.isig[col_edges[0]:col_edges[1], row_edges[0]:row_edges[1]]
        self.sum_intensity = np.asarray(self.image).sum()
        self.mean_intensity = np.mean(np.asarray(self.image))
        self.max_intensity = np.max(np.asarray(self.image))

        self.atom_position = None
        self.atom_position_roi = None

        self.atom_type = None
