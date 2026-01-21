import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.patches import Rectangle
from sklearn.ensemble import IsolationForest
import matplotlib.colors as mcolors
from matplotlib import cm

class CropClassification:
    def _get_intensity(self, atom, metric):
        """
        A Wrapper to return the requested intensity for an atom.
        
        Args:
            atom: An atom object containing intensity attributes.
            metric: A string specifying which intensity to return. One of:
                'mean' | 'max' | 'sum' 
        """

        if isinstance(metric, str):
            attr_map = {
                'mean': 'mean_intensity',
                'max' : 'max_intensity',
                'sum' : 'sum_intensity',
            }
            if metric not in attr_map:
                if getattr(atom, metric) is None:
                    raise ValueError(f"Unknown intensity metric: {metric!r}.")
                else:
                    # print(f"Warning: not a default metric: {metric!r}. Attempting to access attribute directly.")
                    return float(getattr(atom, metric))
            return float(getattr(atom, attr_map[metric]))
        
        raise TypeError("Unknown intensity metric.")
    
    def _get_position(self, atom):
        """
        A Wrapper to return the requested position for an atom.
        
        Args:
            atom: An atom object containing position attributes.
        """

        return atom.atom_position_roi

    # Bayes Prediction:
    # P(d_j|M) = P(M|d_j)P(d_j)/Sum_i(P(M|d_i)P(d_i))
    # P(d|M) = Sum_i(P(d_i|M))
    # P(M|d_i) = Product_k(P(M_k|d_i))
    #
    # Parameters:
    # d = defect existance
    # d_i = defect existance at layer i
    # M = combined measured features
    # M_i = measured features i (e.g. intensity, displacement)

    # Features construction

    def set_nn_coords(self, dis=None, djs=None):
        """
        Set nearest neighbor matrix offsets to dis and djs.
        dis and djs should have the same length.
        dis represents row offsets, and djs represents column offsets.
        e.g. left neighbor: di = -1, dj = 0
        e.g. up neighbor: di = 0, dj = -1

        Args:
            dis: List of row offsets. Default: [0, 0, -1, 1]
            djs: List of column offsets. Default: [-1, 1, 0, 0]
        """
        if dis is None or djs is None:
            self.dis = [0, 0, -1, 1]
            self.djs = [-1, 1, 0, 0]
            return
        if len(dis) != len(djs):
            raise ValueError("dis and djs must have the same length.")
        self.dis = dis
        self.djs = djs

    def get_nn_intensities(self, metric = 'mean'):
        """
        Get nearest neighbor intensity differences from up, down, left, and right. 
        Please override this method if you want a different construction.

        Args:
            metric: A string specifying which intensity to use. One of:
                'mean' | 'max' | 'sum' 

        Results:
            patch.nn_intensity_differences: np.ndarray of shape (n,) representing the intensity differences 
                of the vincinity. Default with [up, down, left, right] neighbors.

        """
        H, W = self.grid_shape
        # Neighboring atoms: up, down, left, right
        dis = self.dis if hasattr(self, 'dis') else [0, 0, -1, 1]
        djs = self.djs if hasattr(self, 'djs') else [-1, 1, 0, 0]
        neighbor_offsets = list(zip(dis, djs))

        for patch in self.grid.values():
            i, j = patch.index

            # Pre-calculate neighbor coordinates
            neighbors = [(i + di, j + dj) for di, dj in neighbor_offsets]

            # Check if ANY neighbor is out of bounds
            if any(not (0 <= ni < H and 0 <= nj < W) for ni, nj in neighbors):
                patch.nn_intensity_differences = None
                continue

            # Calculate
            current_intensity = self._get_intensity(patch, metric)
            patch.nn_intensity_differences = np.array([
                self._get_intensity(self.grid[ni, nj], metric) - current_intensity
                for ni, nj in neighbors
            ])

        return None

    def get_nn_displacements(self):
        """
        Get nearest neighbor intensity differences from up, down, left, and right. 
        Please override this method if you want a different construction.

        Results:
            patch.nn_displacement_differences: np.ndarray of shape (n,) representing the displacement differences 
                of the vincinity. Default with [up, down, left, right] neighbors.
        """
        H, W = self.grid_shape
        # Neighboring atoms: up, down, left, right
        dis = self.dis if hasattr(self, 'dis') else [0, 0, -1, 1]
        djs = self.djs if hasattr(self, 'djs') else [-1, 1, 0, 0]
        neighbor_offsets = list(zip(dis, djs))

        for patch in self.grid.values():
            i, j = patch.index

            # Pre-calculate neighbor coordinates
            neighbors = [(i + di, j + dj) for di, dj in neighbor_offsets]

            # Check if ANY neighbor is out of bounds
            if any(not (0 <= ni < H and 0 <= nj < W) for ni, nj in neighbors):
                patch.nn_displacement_differences = None
                continue

            # Calculate
            current_position = self._get_position(patch)
            patch.nn_displacement_differences = np.array([
                self._get_position(self.grid[ni, nj]) - current_position
                for ni, nj in neighbors
            ])

        return None


    # Legacy Methods for Outlier Detection. Could be used as references.
    # TODO: organize these methods.
    def get_intensity_z_score_outliers(self, outlier_bar = 2, atom_type = 'Lu'):
        """
        Find the outliers by looking at mean intensity differences.

        Args:
            outlier_bar: z values
            atom_type: 'Lu', 'Fe', or 'all'
        """
        means = []
        indices = []

        for patch in self.grid.values():
            if patch.atom_type == atom_type or atom_type == 'all':
                if patch.nn_intensity_differences is not None:
                    mean_val = np.nanmean(patch.nn_intensity_differences)
                else:
                    continue
                means.append(mean_val)
                indices.append(patch.index)

        means = np.array(means)
        self.intensity_from_vincinity_mean = means

        # Find outliers using z-score method
        z_scores = (means - np.mean(means)) / np.std(means)
        self.intensity_from_vincinity_z_scores = z_scores
        self.intensity_from_vincinity_indices = indices
        self.intensity_from_vincinity_outliers_above = [indices[idx] for idx, z in enumerate(z_scores) if z > outlier_bar]  # threshold z > bar
        self.intensity_from_vincinity_outliers_below = [indices[idx] for idx, z in enumerate(z_scores) if z < -outlier_bar] # threshold z < -bar



            
    # Plotting
    def plot_intensity_outliers_z_score_histogram(self, outlier_bar = 2, fit_gaussian = False):
        plt.figure(figsize=(10, 6))
        counts, bins, _ = plt.hist(self.intensity_from_vincinity_mean, bins=40, color='red', alpha=0.5)
        mean_val = np.mean(self.intensity_from_vincinity_mean)
        std_val = np.std(self.intensity_from_vincinity_mean)

        plt.axvline(mean_val, color='black', linestyle='--', linewidth=2, label='Mean')
        plt.axvline(mean_val + outlier_bar*std_val, color='green', linestyle=':', linewidth=2, label=f"+{outlier_bar}σ")
        plt.axvline(mean_val - outlier_bar*std_val, color='green', linestyle=':', linewidth=2, label=f"-{outlier_bar}σ")
        plt.xlabel('Intensity')
        plt.ylabel('Count')

        if fit_gaussian:
            data = np.asarray(self.intensity_from_vincinity_mean)
            mu, sigma = stats.norm.fit(data)
            bin_width = bins[1] - bins[0]
            
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 1000)
            p = stats.norm.pdf(x, mu, sigma)
            
            # Scale to match histogram bin counts
            p_scaled = p * len(data) * bin_width
            plt.plot(x, p_scaled, 'k', linewidth=2, label='Fitted Gaussian', c='blue')

        plt.title('Histogram of Atom Mean Intensity Differences')
        plt.legend()
        plt.show()

    def plot_intensity_outliers_z_score_overlay(self, outlier_bar = 2):
        plt.close()
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.roi)
        above_color='#FF7F0E'
        below_color='#D62728' 

        def _get_edges(patch):
            col_edges = patch.roi_col_edges
            row_edges = patch.roi_row_edges
            x0, x1 = float(col_edges[0]), float(col_edges[1])
            y0, y1 = float(row_edges[0]), float(row_edges[1])
            return x0, x1, y0, y1

        def _draw_box(patch, color):
            edges = _get_edges(patch)
            if edges is None:
                return
            x0, x1, y0, y1 = edges
            wr = x1 - x0
            hr = y1 - y0

            # Draw box (semi-transparent fill to see the atom)
            rect = Rectangle((x0, y0), wr, hr, edgecolor=color, fill=False, zorder = 3, lw = 2)
            ax.add_patch(rect)

        # Draw ABOVE
        for (i, j) in self.intensity_from_vincinity_outliers_above:
            patch = self.grid[i, j]
            _draw_box(patch, color=above_color)

        # Draw BELOW
        for (i, j) in self.intensity_from_vincinity_outliers_below:
            patch = self.grid[i, j]
            _draw_box(patch, color=below_color)

        above_proxy  = Rectangle((0, 0), 1, 1, edgecolor=above_color,  facecolor=above_color)
        below_proxy = Rectangle((0, 0), 1, 1, edgecolor=below_color, facecolor=below_color)
        ax.legend([above_proxy, below_proxy], [f"> +{outlier_bar}σ", f"< -{outlier_bar}σ"], loc="best")

        ax.set_title('ROI with Outlier Atoms (boxes & centers)')
        plt.tight_layout()
        plt.show()

    def plot_intensity_z_score_heatmap(self):
        plt.close()
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Display the base image
        im = ax.imshow(self.roi, cmap='gray')
        
        # 1. Setup Normalization: Centers the colormap at 0 (the mean)
        # Adjust vmin/vmax to control the sensitivity of the colors
        abs_max = np.max(np.abs(self.intensity_from_vincinity_z_scores))
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-abs_max, vmax=abs_max)
        colormap = cm.get_cmap('RdBu_r') # Red (low) to Blue (high)

        def _get_edges(patch):
            x0, x1 = map(float, patch.roi_col_edges)
            y0, y1 = map(float, patch.roi_row_edges)
            return x0, x1, y0, y1

        # 2. Iterate through the entire grid
        for idx, (i, j) in enumerate(self.intensity_from_vincinity_indices):
            patch = self.grid[i, j]
            score = self.intensity_from_vincinity_z_scores[idx]
            
            x0, x1, y0, y1 = _get_edges(patch)
            color = colormap(norm(score))

            # Draw box with a slight transparency (alpha) to see the ROI underneath
            rect = Rectangle((x0, y0), x1 - x0, y1 - y0, 
                            edgecolor=color, 
                            facecolor=color, 
                            alpha=0.3, 
                            lw=1.5)
            ax.add_patch(rect)

        # 3. Add Colorbar to explain the sigmas
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Z-Score (σ from mean)', rotation=270, labelpad=15)

        ax.set_title('ROI Overlay: Intensity Z-Scores')
        plt.tight_layout()
        plt.show()