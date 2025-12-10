import numpy as np
import matplotlib.pyplot as plt

class Patch():

    def __init__(self, crop, index, col_edges, row_edges):
        """
        Initialize a Patch object.
        
        Args:
            crop: A Crop object containing the image data and metadata.
            index: A tuple (i, j) representing the position of the patch in the grid.
            col_edges: A list or array with 2 pixel index of column edges defining the patch boundaries.
            row_edges: A list or array with 2 pixel index of row edges defining the patch boundaries.

        Attributes:
            crop: The Crop object associated with this patch.
            index: The (i, j) index of the patch in the grid.
            col_edges: The column edges of the patch in the full image.
            row_edges: The row edges of the patch in the full image.
            roi_col_edges: The column edges of the patch relative to the crop ROI.
            roi_row_edges: The row edges of the patch relative to the crop ROI.
            width: The width of the patch in pixels.
            height: The height of the patch in pixels.
            image: The 2D numpy array representing the patch image data.
            
            sum_intensity: The sum of pixel intensities in the patch.
            mean_intensity: The mean pixel intensity in the patch.
            max_intensity: The maximum pixel intensity in the patch.
            
            atom_position: The (x, y) position of an atom in the full image (if any).
            atom_position_roi: The (x, y) position of an atom relative to the crop ROI (if any).
            atom_type: The type of atom present in the patch (if any).
        """

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