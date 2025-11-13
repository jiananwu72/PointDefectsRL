import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.patches import Rectangle
from sklearn.ensemble import IsolationForest

class CropClassification:
    def _get_intensity(self, atom, metric):
        """
        Return the requested intensity for an atom.

        metric: 'mean' | 'max' | 'sum' | 'eels_Fe' | 'eels_Lu' | 'voronoi_mean' | 'voronoi_max' | 'voronoi_sum'
        """

        if isinstance(metric, str):
            attr_map = {
                'mean': 'mean_intensity',
                'max' : 'max_intensity',
                'sum' : 'sum_intensity',

                'eels_Fe': 'integrated_intensity_Fe',
                'eels_Lu': 'integrated_intensity_Lu',

                'voronoi_mean': 'mean_intensity_voronoi',
                'voronoi_max' : 'max_intensity_voronoi',
                'voronoi_sum' : 'sum_intensity_voronoi',
            }
            if metric not in attr_map:
                raise ValueError(f"Unknown intensity metric: {metric!r}.")
            return float(getattr(atom, attr_map[metric]))
        
        raise TypeError("Unknown intensity metric.")

    def get_nn_intensities_same(self, number_of_atoms_from_edge = 2,
                                horizonal_radius = 2, vertical_radius = 2, metric = 'mean'):
        """
        Get nearest neighbor intensity for same types of atoms.

        number_of_atoms_from_edge: number of atoms that will avoid counting from the edge and will be labeled None.

        horizontal_radius and vertical_radius: number of atoms to be considered around in the grid
        will not count atoms of different types inside the radius.
        """
        for patch in self.grid.values():
            # Get rid of edge atoms
            if patch.index[0] < number_of_atoms_from_edge or patch.index[0] >= self.grid_shape[0] - number_of_atoms_from_edge:
                patch.nn_same_atom_intensity_differences = None
                continue
            if patch.index[1] < number_of_atoms_from_edge or patch.index[1] >= self.grid_shape[1] - number_of_atoms_from_edge:
                patch.nn_same_atom_intensity_differences = None
                continue

            i = patch.index[0]
            j = patch.index[1]

            center_intensity = self._get_intensity(patch, metric)
            center_type = getattr(patch, 'atom_type', None)
            same_cols = []
            for di in range(-horizonal_radius, horizonal_radius + 1):
                col = []
                for dj in range(-vertical_radius, vertical_radius + 1):
                    ni, nj = i + di, j + dj
                    try:
                        neigh = self.grid[ni, nj]
                        if getattr(neigh, 'atom_type', None) == center_type:
                            col.append(self._get_intensity(neigh, metric))
                        else:
                            col.append(np.nan)  # keep shape; not same type
                    except KeyError:
                        col.append(np.nan)      # out of bounds
                same_cols.append(col)
            same_atom_intensities = np.array(same_cols, dtype=float)
            patch.nn_same_atom_intensity_differences = center_intensity - same_atom_intensities

    def get_nn_intensities_all(self, number_of_atoms_from_edge = 2,
                                horizonal_radius = 1, vertical_radius = 1, metric = 'mean'):
        """
        Get nearest neighbor intensity for all types of atoms.

        number_of_atoms_from_edge: number of atoms that will avoid counting from the edge and will be labeled None.
        horizontal_radius and vertical_radius: number of atoms to be considered around in the grid
        """
        for patch in self.grid.values():
            # Get rid of edge atoms
            if patch.index[0] < number_of_atoms_from_edge or patch.index[0] >= self.grid_shape[0] - number_of_atoms_from_edge:
                patch.nn_all_atom_intensity_differences = None
                continue
            if patch.index[1] < number_of_atoms_from_edge or patch.index[1] >= self.grid_shape[1] - number_of_atoms_from_edge:
                patch.nn_all_atom_intensity_differences = None
                continue

            i = patch.index[0]
            j = patch.index[1]

            center_intensity = self._get_intensity(patch, metric)
            cols = []
            for di in range(-horizonal_radius, horizonal_radius + 1):
                col = []
                for dj in range(-vertical_radius, vertical_radius + 1):
                    ni, nj = i + di, j + dj
                    try:
                        neigh = self.grid[ni, nj]
                        col.append(self._get_intensity(neigh, metric))
                    except KeyError:
                        col.append(np.nan)      # out of bounds
                cols.append(col)
            atom_intensities = np.array(cols, dtype=float)
            patch.nn_all_atom_intensity_differences = center_intensity - atom_intensities

    def get_nn_displacements(self, number_of_atoms_from_edge = 2):
        for patch in self.grid.values():
            # Get rid of edge atoms
            if patch.index[0] < number_of_atoms_from_edge or patch.index[0] >= self.grid_shape[0] - number_of_atoms_from_edge:
                patch.nn_displacements = None
                continue
            if patch.index[1] < number_of_atoms_from_edge or patch.index[1] >= self.grid_shape[1] - number_of_atoms_from_edge:
                patch.nn_displacements = None
                continue

            i = patch.index[0]
            j = patch.index[1]

            # Displacement differences for 9 atoms around
            current_position = patch.atom_position_roi
            dis = (-1, 0, 1)
            djs = (-1, 0, 1)
            patch.nn_displacements = np.array([
                [np.linalg.norm(self.grid[i+di, j+dj].atom_position_roi-current_position)
                                for dj in djs] for di in dis])
            

    # Finding intensity outliers by checking z_scores
    def get_intensity_z_score_outliers(self, outlier_bar = 2, atom_type = 'Lu', atom_selection = 'same'):
        """
        Find the outliers by looking at mean intensity differences.

        outlier_bar: z values
        atom_type: 'Lu', 'Fe', or 'all'
        atom_selection: 'same' or 'all'
        """
        means = []
        indices = []

        for patch in self.grid.values():
            if patch.atom_type == atom_type or atom_type == 'all':
                if atom_selection == 'same' and patch.nn_same_atom_intensity_differences is not None:
                    mean_val = np.nanmean(patch.nn_same_atom_intensity_differences)
                elif atom_selection == 'all' and patch.nn_all_atom_intensity_differences is not None:
                    mean_val = np.nanmean(patch.nn_all_atom_intensity_differences)
                else:
                    continue
                means.append(mean_val)
                indices.append(patch.index)

        means = np.array(means)
        self.intensity_from_vincinity_mean = means

        # Find outliers using z-score method
        z_scores = (means - np.mean(means)) / np.std(means)
        self.intensity_from_vincinity_z_scores = z_scores
        self.intensity_from_vincinity_outliers_above = [indices[idx] for idx, z in enumerate(z_scores) if z > outlier_bar]  # threshold z > bar
        self.intensity_from_vincinity_outliers_below = [indices[idx] for idx, z in enumerate(z_scores) if z < -outlier_bar] # threshold z < -bar

    # Finding outliers using Isolation Forest
    def get_intensity_outliers_isoforest(self, contamination=0.05, random_state=0, atom_selection = 'same', atom_type = 'Lu'):
        """
        Detect outliers among atoms using Isolation Forest on the vector
        from nn_same_atom_intensity_differences. Also classify
        outliers as 'too high' or 'too low' based on the mean of that array.

        Writes:
        - self.iforest_anomaly_score : np.ndarray (aligned to collected indices)
        - self.iforest_outliers      : list[(i,j)]
        - self.iforest_outliers_high : list[(i,j)]  # mean above median -> "too high"
        - self.iforest_outliers_low  : list[(i,j)]  # mean below median -> "too low"
        - self.iforest_indices       : list[(i,j)]  # order matches scores
        - self.iforest_feature_medians: np.ndarray (K,) used to fill NaNs
        Returns:
        dict with keys: indices, scores, outliers, outliers_high, outliers_low
        """
        # 1) collect patches with valid same-diff arrays
        indices = []
        rows = []
        means_1d = []   # store mean of K elements for later "high/low" classification (before imputation)
        for (i, j), patch in self.grid.items():
            arr = None
            if atom_selection == 'same':
                if patch.atom_type == atom_type or atom_type == 'all':
                    arr = patch.nn_same_atom_intensity_differences
            elif atom_selection == 'all':
                arr = patch.nn_all_atom_intensity_differences
            
            if arr is not None:
                arr = np.asarray(arr, dtype=float)
                indices.append((i, j))
                rows.append(arr.reshape(-1))                 # K-D
                means_1d.append(np.nanmean(arr))             # NaN-safe mean for direction

        X = np.vstack(rows)          # (N, K)
        means_1d = np.asarray(means_1d, dtype=float)

        # 2) feature-wise median imputation for NaNs (keeps each feature’s typical scale)
        feat_medians = np.nanmedian(X, axis=0)
        nan_mask = ~np.isfinite(X)
        if np.any(nan_mask):
            X = np.where(nan_mask, feat_medians, X)

        # 3) Isolation Forest fit (higher decision_function = more normal)
        iso = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        ).fit(X)

        decision = iso.decision_function(X)   # higher => more normal
        anomaly_score = -decision             # higher => more anomalous (convenient)

        # label: -1 for outliers
        labels = iso.predict(X)
        outlier_mask = (labels == -1)

        # 4) Classify outliers as "too high" vs "too low"
        #    Use robust center of the per-patch mean (median).
        center = np.nanmedian(means_1d)
        high_mask = outlier_mask & (means_1d >= center)
        low_mask  = outlier_mask & (means_1d <  center)

        outliers       = [indices[k] for k in np.where(outlier_mask)[0]]
        outliers_high  = [indices[k] for k in np.where(high_mask)[0]]
        outliers_low   = [indices[k] for k in np.where(low_mask)[0]]

        # 5) persist for later inspection/plotting
        self.iforest_anomaly_score = anomaly_score
        self.iforest_outliers = outliers
        self.iforest_outliers_high = outliers_high
        self.iforest_outliers_low = outliers_low
        self.iforest_indices = indices
        self.iforest_feature_medians = feat_medians

        # Optionally, write flags back to each patch
        for (idx, (i, j)) in enumerate(indices):
            patch = self.grid[i, j]
            patch.iforest_anomaly_score = float(anomaly_score[idx])
            patch.iforest_is_outlier = bool(outlier_mask[idx])
            if high_mask[idx]:
                patch.iforest_direction = 'high'
            elif low_mask[idx]:
                patch.iforest_direction = 'low'
            else:
                patch.iforest_direction = 'inlier'

        return {
            "indices": indices,
            "scores": anomaly_score,
            "outliers": outliers,
            "outliers_high": outliers_high,
            "outliers_low": outliers_low
        }

            
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