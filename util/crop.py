import numpy as np
import scipy
import matplotlib.pyplot as plt
import atomap.api as am

from .crop_intensity import CropIntensity
from .crop_classification import CropClassification
from .patch import Patch

class Crop(CropIntensity, CropClassification):

    def __init__(self, signal, left = None, right = None, start = None, end = None):
        self.signal = signal

        if left is None: left = 0
        if right is None: right = signal.data.shape[1]
        if start is None: start = 0
        if end is None: end = signal.data.shape[0]
        self.left = left
        self.right = right
        self.start = start
        self.end = end

        self.roi = self.signal.isig[left:right, start:end]

        self.vertical_peaks = None
        self.vertical_troughs = None
        self.x_axis_scan_strengths = None

        self.horizontal_peaks = None
        self.horizontal_troughs = None
        self.y_axis_scan_strengths = None

        self.grid = None
        self.grid_shape = None
        self.atom_positions = None

    def get_vertical_peaks(self, left=None, right=None, get_plot=False, 
                           detect_height=1, min_sep=3, prom_coeff = 0.1):
        
        H, W = self.signal.data.shape
        if left is None:  left = self.left
        if right is None: right = self.right
        if right > W:
            right = W
            print("right exceeds image width; using image width instead.")

        # Build strengths for y in [start, end)
        strengths = []
        for i in range(self.start, self.end):
        # for i in range(0, H):  # use full image coordinates for consistency
            s_cropped = self.signal.isig[left:right, i:i+detect_height]
            strengths.append(np.asarray(s_cropped).sum())
        strengths = np.asarray(strengths)

        # Detect peaks in the local (0..N-1) index space
        prom = prom_coeff * (strengths.max() - strengths.min())        
        peaks, _   = scipy.signal.find_peaks( strengths, distance=min_sep, prominence=prom)
        troughs, _ = scipy.signal.find_peaks(-strengths, distance=min_sep, prominence=prom)

        # Convert to GLOBAL y-indices once
        y_coords   = np.arange(self.start, self.end)
        # y_coords   = np.arange(0, H)  # use full image coordinates for consistency
        self.horizontal_peaks   = y_coords[peaks]
        self.horizontal_troughs = y_coords[troughs]
        self.y_axis_scan_strengths = strengths

        if get_plot:
            plt.figure()
            plt.plot(y_coords, strengths, '-k', label='strip strength')
            plt.plot(self.horizontal_peaks,   strengths[peaks],   'ro', label='local maxima')
            plt.plot(self.horizontal_troughs, strengths[troughs], 'go', label='local minima')
            plt.xlabel('y-axis pixel index');  plt.ylabel('Summed intensity')
            plt.title('Vertical window strengths with detected peaks and troughs')
            plt.legend()
            # plt.xlim(y_coords[0], y_coords[-1])  # matches data coordinates
            plt.tight_layout(); plt.show()

    def get_horizontal_peaks(self, start=None, end=None, get_plot=False, 
                             detect_width=1, min_sep=3, prom_coeff = 0.1):
        
        H, W = self.signal.data.shape
        if start is None: start = self.start
        if end   is None: end = self.end
        if end > H:
            end = H
            print("end exceeds image height; using image height instead.")

        # Build strengths for x in [left, right)
        strengths = []
        for i in range(self.left, self.right):
            s_cropped = self.signal.isig[i:i+detect_width, start:end]
            strengths.append(np.asarray(s_cropped).sum())
        strengths = np.asarray(strengths)

        # Detect peaks in the local (0..N-1) index space
        prom = prom_coeff * (strengths.max() - strengths.min())        
        peaks, _   = scipy.signal.find_peaks( strengths, distance=min_sep, prominence=prom)
        troughs, _ = scipy.signal.find_peaks(-strengths, distance=min_sep, prominence=prom)

        # Convert to GLOBAL x-indices once
        x_coords   = np.arange(self.left, self.right)
        self.vertical_peaks   = x_coords[peaks]
        self.vertical_troughs = x_coords[troughs]
        self.x_axis_scan_strengths = strengths

        if get_plot:
            plt.figure()
            plt.plot(x_coords, strengths, '-k', label='strip strength')
            plt.plot(self.vertical_peaks,   strengths[peaks],   'ro', label='local maxima')
            plt.plot(self.vertical_troughs, strengths[troughs], 'go', label='local minima')
            plt.xlabel('x-axis pixel index');  plt.ylabel('Summed intensity')
            plt.title('Horizontal window strengths with detected peaks and troughs')
            plt.legend()
            plt.xlim(x_coords[0], x_coords[-1])  # matches data coordinates
            plt.tight_layout(); plt.show()

    def plot_grid_troughs(self):
        plt.close()
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.roi)

        # Overlay grid lines at detected troughs
        for x in self.vertical_troughs:
            ax.axvline(x-self.left, color='magenta', linestyle='--', linewidth=1)
        for y in self.horizontal_troughs:
            ax.axhline(y-self.start, color='cyan', linestyle='--', linewidth=1)

        ax.set_title('ROI with Detected Grid (Troughs)')
        plt.tight_layout()
        plt.show()


    def build_grid_dict(self):
        """
        Build a 2D dictionary of grid cells bounded by adjacent horizontal/vertical troughs.
        Uses the *first through last* entries of self.vertical_troughs and self.horizontal_troughs.

        Returns
        -------
        grid : dict of patches
        """

        import numpy as np

        # Require troughs to be present
        if not hasattr(self, "vertical_troughs") or self.vertical_troughs is None:
            raise ValueError("vertical_troughs not set. Run get_vertical_peaks(...).")
        if not hasattr(self, "horizontal_troughs") or self.horizontal_troughs is None:
            raise ValueError("horizontal_troughs not set. Run get_horizontal_peaks(...).")

        W, H = self.signal.data.shape  # your convention: (x = width, y = height)

        # Prepare sorted, unique, in-bounds edges
        v = np.asarray(self.vertical_troughs, dtype=int)
        h = np.asarray(self.horizontal_troughs, dtype=int)

        # Must have at least two edges in each direction
        if v.size < 2 or h.size < 2:
            raise ValueError("Need at least two vertical and two horizontal troughs to define a grid.")

        # Make adjacent edge pairs (first..last is implicit by iterating adjacent pairs)
        col_edges = [(int(v[i]), int(v[i+1])) for i in range(len(v)-1)]
        row_edges = [(int(h[j]), int(h[j+1])) for j in range(len(h)-1)]

        grid = {}
        col_idx = 0
        for (x0, x1) in col_edges:
            row_idx = 0
            for (y0, y1) in row_edges:
                patch = Patch(self, (col_idx, row_idx), (x0, x1), (y0, y1))
                grid[(col_idx, row_idx)] = patch
                row_idx += 1
            col_idx += 1

        self.grid = grid
        self.grid_shape = (len(col_edges), len(row_edges))

    def plot_grid_intensities(self, intensity_type = 'mean'):
        """
        intensity_type: 'mean', 'sum', or 'max'
        """

        x_len = self.grid_shape[0]
        y_len = self.grid_shape[1]

        # Build 2D array of intensity
        intensity_map = np.zeros((x_len, y_len))
        for i in range(x_len):
            for j in range(y_len):
                if intensity_type == 'mean':
                    intensity_map[i, j] = self.grid[i, j].mean_intensity
                elif intensity_type == 'max':
                    intensity_map[i, j] = self.grid[i, j].max_intensity
                elif intensity_type == 'sum':
                    intensity_map[i, j] = self.grid[i, j].sum_intensity
                else:
                    raise ValueError("Invalid intensity_type string")

        # Plot: y axis is from up to down, so no need to flip
        plt.figure(figsize=(6, 5))
        plt.imshow(intensity_map.T, origin='upper', aspect='auto', cmap='viridis')
        plt.xlabel('i (x, left to right)')
        plt.ylabel('j (y, up to down)')
        plt.title('2D Mapping of '+intensity_type+' intensity')
        plt.colorbar(label='intensity')
        plt.show()

    def get_atom_positions(self, mask_size = 10, mask_std = 3, get_plot = False):
        """
        separation: separation for atomap's atom finding
        sigma_allowance: std difference for identifying outliers
        """
        # Create a kernel
        g1d = scipy.signal.windows.gaussian(mask_size, std=mask_std)   # 1D Gaussian
        kernel = np.outer(g1d, g1d)

        
        atom_positions = {}
        for patch in self.grid.values():
            # Convolve patch image with circular kernel
            conv_img = scipy.ndimage.convolve(patch.image.data, kernel, mode='constant', cval=0.0)
            max_idx = np.unravel_index(np.argmax(conv_img), conv_img.shape)
            global_x = patch.roi_col_edges[0] + max_idx[1]
            global_y = patch.roi_row_edges[0] + max_idx[0]

            patch.atom_position = np.array([max_idx[1], max_idx[0]])
            patch.atom_position_roi = np.array([global_x, global_y])
            atom_positions[patch.index[0], patch.index[1]] = patch.atom_position_roi
        self.atom_positions = atom_positions
        positions_array = np.array(list(self.atom_positions.values()))

        if get_plot:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(self.roi)
            for x in self.vertical_troughs:
                ax.axvline(x-self.left, color='magenta', linestyle='--', linewidth=1)
            for y in self.horizontal_troughs:
                ax.axhline(y-self.start, color='cyan', linestyle='--', linewidth=1)
            ax.scatter(positions_array[:,0], positions_array[:,1], s=4, c='r')
            plt.title("Initially Detected Atom Positions by Convolving a Gaussian with Troughs")
            plt.show()


    def refine_atom_positions(self, mask_radius = None, percent_to_nn=None, get_plot = False):
        """
        refine_atom_positions
        """

        keys = list(self.atom_positions.keys())
        positions_array = np.array(list(self.atom_positions.values()))

        sublattice = am.Sublattice(self.atom_positions, image=self.roi)
        sublattice.find_nearest_neighbors()
        kwargs = {"mask_radius": mask_radius}
        if mask_radius is None:  # only override if explicitly set
            kwargs["percent_to_nn"] = percent_to_nn
        sublattice.refine_atom_positions_using_center_of_mass(**kwargs)
        sublattice.refine_atom_positions_using_2d_gaussian(**kwargs)

        self.atom_positions = {k: v for k, v in zip(keys, positions_array)}

        for patch in self.grid.values():
            patch.atom_position_roi = self.atom_positions[patch.index[0], patch.index[1]]
            patch.atom_position = np.array([patch.atom_position_roi[0] - patch.roi_col_edges[0],
                                            patch.atom_position_roi[1] - patch.roi_row_edges[0]])

        if get_plot:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(self.roi)
            for x in self.vertical_troughs:
                ax.axvline(x-self.left, color='magenta', linestyle='--', linewidth=1)
            for y in self.horizontal_troughs:
                ax.axhline(y-self.start, color='cyan', linestyle='--', linewidth=1)
            ax.scatter(positions_array[:,0], positions_array[:,1], s=4, c='r')
            plt.title("Refined Atom Positions with Troughs")
            plt.show()
    
    def plot_positions(self, show_troughs = False):
        positions_array = np.array(list(self.atom_positions.values()))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.roi)
        if show_troughs:
            for x in self.vertical_troughs:
                ax.axvline(x-self.left, color='magenta', linestyle='--', linewidth=1)
            for y in self.horizontal_troughs:
                ax.axhline(y-self.start, color='cyan', linestyle='--', linewidth=1)
        ax.scatter(positions_array[:,0], positions_array[:,1], s=4, c='r')
        plt.title("Atom Positions")
        plt.show()

    
    def get_atom_types(self, tol = 0.0):
        """
        Only For LuFeO3. Not for generalization
        """
        layer_intensities = np.zeros(self.grid_shape[1])

        for patch in self.grid.values():
            layer_intensities[patch.index[1]] += patch.mean_intensity
        
        n_layers = layer_intensities.shape[0]
        is_lu_layer = np.zeros(n_layers, dtype=bool)

        # Decide Lu vs Fe by local (one-neighbor) comparison
        for j in range(n_layers):
            left = layer_intensities[j - 1] if j - 1 >= 0 else -np.inf
            right = layer_intensities[j + 1] if j + 1 < n_layers else -np.inf
            if layer_intensities[j] > max(left, right) + tol:
                is_lu_layer[j] = True

        # Assign atom_type to each patch based on its layer
        for patch in self.grid.values():
            j = patch.index[1]
            patch.atom_type = 'Lu' if is_lu_layer[j] else 'Fe'


