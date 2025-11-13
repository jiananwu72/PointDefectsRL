import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from skimage.segmentation import find_boundaries

class CropIntensity:

    def voronoi(self, get_plot = False):
        keys = list(self.grid.keys())
        img = self.roi.data
        H, W = img.shape
        pts_rc = np.array(list(self.atom_positions.values()))
        N = len(pts_rc)

        # 1) Build a marker image with unique IDs at atom pixels
        markers = np.zeros((H, W), dtype=int)
        for i, (c, r) in enumerate(np.round(pts_rc).astype(int), start=1):
            markers[r, c] = i

        # 2) Voronoi
        foreground = markers > 0
        # Distance transform expects "0 := foreground", so invert:
        _, (ri, ci) = ndimage.distance_transform_edt(~foreground, return_indices=True)

        # Pull the ID from the nearest marker for each pixel; pixels that *are* markers use themselves
        labels = markers[ri, ci]

        # 3) Per-atom intensity stats
        indices = np.arange(1, N+1)  # ignore label 0
        sum_intensity   = ndimage.sum(img, labels=labels, index=indices)
        sum_intensity_dict = {k: v for k, v in zip(keys, sum_intensity)}
        mean_intensity  = ndimage.mean(img, labels=labels, index=indices)
        mean_intensity_dict = {k: v for k, v in zip(keys, mean_intensity)}
        max_intensity = ndimage.maximum(img, labels=labels, index=indices)
        max_intensity_dict = {k: v for k, v in zip(keys, max_intensity)}

        xs, ys = zip(*keys)         # unpack first and second elements from each key
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        for patch in self.grid.values():
            x = patch.index[0]
            y = patch.index[1]
            if x == x_min or x == x_max or y == y_min or y == y_max:
                patch.sum_intensity_voronoi = None
                patch.mean_intensity_voronoi = None
                patch.max_intensity_voronoi = None
            else:
                patch.sum_intensity_voronoi = sum_intensity_dict[x, y]
                patch.mean_intensity_voronoi = mean_intensity_dict[x, y]
                patch.max_intensity_voronoi = max_intensity_dict[x, y]

        if get_plot:
            # boundaries: boolean mask of cell edges
            bnd = find_boundaries(labels, mode="thick")

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(img, cmap='gray')                     # STEM image
            cmap = ListedColormap([(0, 0, 0, 0), (0, 1, 1, 1)])
            ax.imshow(bnd, cmap=cmap, vmin=0, vmax=1, alpha=0.3, interpolation='nearest') 
            ax.scatter(pts_rc[:,0], pts_rc[:,1], s=10, c='r')      # atom centers (x=col, y=row)
            ax.set_title("Pixel Voronoi boundaries over image")
            ax.set_axis_off()
            plt.show()

    def foo(self):
        
        return


#     def get_intensities(self, atom_type = 'all', type='rectangle', width_type='nm', width=0.2, height=None, start_ind = 0, end_ind = np.inf):
#         """
#         Get the intensity profiles for all layers in the crop and store them in each layer.

#         atom_type: str
#             The type of atoms to consider for intensity calculation.
#             ('all', 'Fe', 'Lu')
#         """
#         if end_ind > len(self.layers):
#             end_ind = len(self.layers)

#         if height is None:
#             height = width

#         for layer_ind in range(start_ind, end_ind):
#             layer = self.layers[layer_ind]
#             if atom_type == 'all' or layer.atom_type == atom_type:
#                 layer.get_intensity(type=type, width_type=width_type, width=width, height=height)

#         self.intensity_type = type
#         if width_type == 'nm':
#             scale = self.signal.axes_manager['x'].scale
#             width = int(width / scale)
#             height = int(height / scale)
#         elif width_type == 'pixels':
#             width = int(width)
#             height = int(height)
#         else:
#             raise ValueError(f"Unknown width type: {width_type}")
    
#         if type == 'verticle_bar':
#                 height = self.window_height
        
#         self.width = width
#         self.height = height
    
#     def plot_intensities_verticle(self, x_pixel, radius = 10, atom_type = 'all'):
#         """
#         Plot the mean intensity profiles for a verticle area in the crop.

#         atom_type: str
#             The type of atoms to consider for intensity calculation.
#             ('all', 'Fe', 'Lu')
#         """

#         fig, ax = plt.subplots(1, 2, figsize=(9, 10), gridspec_kw={'width_ratios': [1, 3]})

#         boundaries = np.array([max(x_pixel + self.left - radius - self.width, 0),
#                       min(x_pixel + self.left + radius + self.width, self.signal.data.shape[1]),
#                       max(self.start - self.height, 0),
#                       min(self.end + self.height, self.signal.data.shape[0])]).astype(int)
#         print(boundaries)
#         image = self.signal.isig[boundaries[0]:boundaries[1], boundaries[2]:boundaries[3]]

#         # Plot background image
#         ax[0].imshow(image, interpolation='nearest', vmax=np.median(np.array(image))+3*np.std(np.array(image)))

#         for window_ind in range(len(self.layers)):
#             if atom_type == 'all' or self.layers[window_ind].atom_type == atom_type:
#                 atoms = self.layers[window_ind].atom_positions.copy()
#                 in_range_mask = (atoms[:, 1] >= x_pixel - radius) & (atoms[:, 1] <= x_pixel + radius)
#                 row_indices = np.where(in_range_mask)[0]
#                 atom_in_range = atoms[row_indices]

#                 # Plot atoms positions
#                 x = atom_in_range[:, 1] + self.left - boundaries[0]
#                 y = atom_in_range[:, 0] + self.layers[window_ind].pixel_id_extended - boundaries[2]
#                 ax[0].scatter(x, y, s=10, alpha = 0.3, color = 'red')

#                 # Plot intensities integration areas
#                 for i in range(len(row_indices)):
#                     if self.intensity_type == 'rectangle' or self.intensity_type == 'verticle_bar':
#                         corner_x = x - self.layers[window_ind].width // 2
#                         corner_y = y - self.layers[window_ind].height // 2
#                         # Create a rectangle patch
#                         rect = patches.Rectangle(
#                             (corner_x, corner_y), 
#                             self.layers[window_ind].width, 
#                             self.layers[window_ind].height,
#                             linewidth=1, 
#                             alpha=0.7,
#                             edgecolor='white', 
#                             facecolor='none'
#                         )
#                         ax[0].add_patch(rect)

#                     elif self.intensity_type == 'ellipse':
#                         ellipse_center_x = x
#                         ellipse_center_y = y
                        
#                         ell = patches.Ellipse(
#                             (ellipse_center_x, ellipse_center_y),
#                             self.layers[window_ind].width,
#                             self.layers[window_ind].height,
#                             linewidth=1,
#                             alpha=0.7,
#                             edgecolor='white',
#                             facecolor='none'
#                         )
#                         ax[0].add_patch(ell)

#                 # Plot intensities scatter plot
#                 if row_indices is not None and len(self.layers[window_ind].intensities) > 0:
#                     for i in range(len(row_indices)):
#                         intensity = self.layers[window_ind].intensities[row_indices[i]]
#                         print(row_indices[i], intensity)
#                         intensity_x = atoms[row_indices[i], 0] + self.layers[window_ind].image_dimensions[2] - self.start
#                         if self.layers[window_ind].atom_type == 'Fe':
#                             ax[1].scatter(intensity, intensity_x, color='b', alpha=0.7)
#                         elif self.layers[window_ind].atom_type == 'Lu':
#                             ax[1].scatter(intensity, intensity_x, color='r', alpha=0.7)
#                         ax[1].yaxis.set_inverted(True)

#     def plot_intensities(self, alpha = 0.3, atom_type = 'all', min_val = 0, max_val = np.inf, 
#                          type = 'mean', start_ind = 0, end_ind = np.inf):
#         if end_ind > len(self.layers):
#             end_ind = len(self.layers)

#         image = self.signal.isig[self.left:self.right, self.start:self.end + self.height]
#         fig, ax = plt.subplots(figsize=(10,8))
#         ax.imshow(image, cmap='gray', alpha=alpha)

#         intensities_all = []
#         for window_ind in range(len(self.layers)):
#             if self.layers[window_ind].atom_type == atom_type or atom_type == 'all':
#                 if type == 'mean':
#                     intensities_all.extend(self.layers[window_ind].intensities.values())
#                 elif type == 'max':
#                     intensities_all.extend(self.layers[window_ind].max_intensities.values())
#         if intensities_all:
#             vmin = min(intensities_all)
#             vmax = max(intensities_all)

#         for window_ind in range(start_ind, end_ind):
#             if self.layers[window_ind].atom_type == atom_type or atom_type == 'all':
#                 filtered_x = []
#                 filtered_y = []
#                 filtered_intensities = []
#                 if type == 'mean':
#                     intensities = self.layers[window_ind].intensities
#                 elif type == 'max':
#                     intensities = self.layers[window_ind].max_intensities
#                 for i, intensity in enumerate(intensities.values()):
#                     if intensity < max_val and intensity > min_val: 
#                         filtered_x.append(self.layers[window_ind].atom_positions[i,1])
#                         filtered_y.append(self.layers[window_ind].atom_positions[i,0]+ self.layers[window_ind].pixel_id_extended - self.start)
#                         filtered_intensities.append(intensity)

#                 if filtered_x and filtered_y and filtered_intensities:
#                     scatter = ax.scatter(filtered_x, 
#                                 filtered_y,
#                                 s=10,
#                                 c=filtered_intensities,
#                                 cmap='viridis',
#                                 vmin=vmin, vmax=vmax)
            
#         fig.colorbar(scatter, ax=ax, label='Intensities')

#     def plot_intensity_histogram(self, atom_type = 'all', bins=100, type = 'mean'):
#         """
#         Plot the histogram of intensities for all layers in the crop.

#         atom_type: str
#             The type of atoms to consider for intensity calculation.
#             ('all', 'Fe', 'Lu')
#             for 'all', it will plot histogram for both atoms sharing the same x-axis.
#         """

#         if atom_type == 'all':
#             intensities_Fe = []
#             intensities_Lu = []
#             for window_ind in range(len(self.layers)):
#                 if type == 'mean':
#                     values = self.layers[window_ind].intensities.values()
#                 elif type == 'max':
#                     values = self.layers[window_ind].max_intensities.values()

#                 if self.layers[window_ind].atom_type == 'Fe':
#                     intensities_Fe.extend(values)
#                 elif self.layers[window_ind].atom_type == 'Lu':
#                     intensities_Lu.extend(values)

#             plt.figure(figsize=(8,8))
#             plt.hist(intensities_Fe, bins=bins, alpha=0.5, label='Fe', color='b')
#             plt.hist(intensities_Lu, bins=bins, alpha=0.5, label='Lu', color='r')

#             mean_Lu = np.mean(np.array(intensities_Lu))
#             std_Lu = np.std(np.array(intensities_Lu))  
#             mean_Fe = np.mean(np.array(intensities_Fe))
#             std_Fe = np.std(np.array(intensities_Fe))
#             plt.axvline(mean_Lu, color='r', linestyle='dashed', linewidth=2, label='Lu mean = '+f'{mean_Lu:.2f}')
#             plt.axvspan(mean_Lu - std_Lu, mean_Lu + std_Lu, alpha=0.2, color='r', label='Lu std = '+f'{std_Lu:.2f}')
#             plt.axvline(mean_Fe, color='b', linestyle='dashed', linewidth=2, label='Fe mean = '+f'{mean_Fe:.2f}')
#             plt.axvspan(mean_Fe - std_Fe, mean_Fe + std_Fe, alpha=0.2, color='b', label='Fe std = '+f'{std_Fe:.2f}')


#             if type == 'mean':
#                 plt.xlabel('Mean Intensity')
#                 plt.ylabel('Count')
#                 plt.title('Histogram of Mean Intensities')
#             elif type == 'max':
#                 plt.xlabel('Max Intensity')
#                 plt.ylabel('Count')
#                 plt.title('Histogram of Max Intensities')
#             plt.legend()
#         else:
#             intensities_all = []
#             for window_ind in range(len(self.layers)):
#                 if type == 'mean':
#                     values = self.layers[window_ind].intensities.values()
#                 elif type == 'max':
#                     values = self.layers[window_ind].max_intensities.values()

#                 if self.layers[window_ind].atom_type == atom_type:
#                     intensities_all.extend(values)

#             plt.figure(figsize=(8,8))
#             plt.hist(intensities_all, bins=bins)
#             if type == 'mean':
#                 plt.xlabel('Mean Intensity')
#                 plt.ylabel('Count')
#                 plt.title('Histogram of Mean Intensities')
#             elif type == 'max':
#                 plt.xlabel('Max Intensity')
#                 plt.ylabel('Count')
#                 plt.title('Histogram of Max Intensities')

    


#     def plot_intensity_histogram_stacked(self, atom_type = 'all', bins=20, type = 'mean'):
#         """
#         Plot the stacked histogram of intensities for all layers in the crop.

#         atom_type: str
#             The type of atoms to consider for intensity calculation.
#             ('all', 'Fe', 'Lu')
#             for 'all', it will plot histogram for both atoms sharing the same x-axis.
#         """

#         intensities_dict = {}
#         mean_dict = {}
#         std_dict = {}

#         counter = 0
#         for window_ind in range(len(self.layers)):
#             if type == 'mean':
#                 values = self.layers[window_ind].intensities.values()
#                 mean = self.layers[window_ind].mean_intensities
#                 std = self.layers[window_ind].std_intensities
#             elif type == 'max':
#                 values = self.layers[window_ind].max_intensities.values()
#                 mean = self.layers[window_ind].mean_max_intensities
#                 std = self.layers[window_ind].std_max_intensities

#             if self.layers[window_ind].atom_type == atom_type or atom_type == 'all':
#                 intensities_dict[counter] = values
#                 mean_dict[counter] = mean
#                 std_dict[counter] = std
#                 counter += 1
        
#         fig, ax = plt.subplots(
#             nrows=counter, 
#             ncols=1, 
#             figsize=(10, counter),
#             sharex=True, 
#             sharey=True
#         )
#         if counter == 1:
#             ax = [ax]

#         counter = 0
#         for i in range(len(self.layers)):
#             if self.layers[i].atom_type == 'Lu':
#                 ax[counter].hist(intensities_dict[i], bins=bins, alpha=0.7, label='Lu', color='r')
#                 arr = list(intensities_dict[i])
#                 mean = mean_dict[i]
#                 std = std_dict[i]
#                 ax[counter].axvline(mean, color='black', linestyle='dashed', linewidth=2, label='Mean')
#                 ax[counter].axvspan(mean - std, mean + std, alpha=0.2, color='g', label='std')
#                 counter += 1
#             elif self.layers[i].atom_type == 'Fe':
#                 ax[counter].hist(intensities_dict[i], bins=bins, alpha=0.7, label='Fe', color='b')
#                 arr = list(intensities_dict[i])
#                 mean = mean_dict[i]
#                 std = std_dict[i]
#                 ax[counter].axvline(mean, color='black', linestyle='dashed', linewidth=2, label='Mean')
#                 ax[counter].axvspan(mean - std, mean + std, alpha=0.2, color='g', label='std')
#                 counter += 1

#         if type == 'mean':
#             ax[0].set_title('Stacked Histogram of Mean Intensities')
#         elif type == 'max':
#             ax[0].set_title('Stacked Histogram of Max Intensities')
#         fig.subplots_adjust(hspace=0)
#         plt.tight_layout()

#     def plot_intensity_scatter_stacked(self, atom_type = 'all', type = 'mean'):
#         counter_Lu = 0
#         counter_Fe = 0
#         for window_ind in range(len(self.layers)):
#             if self.layers[window_ind].atom_type == 'Lu':
#                 counter_Lu += 1
#             elif self.layers[window_ind].atom_type == 'Fe':
#                 counter_Fe += 1
        
#         if atom_type == 'Lu':
#             counter = counter_Lu
#         elif atom_type == 'Fe':
#             counter = counter_Fe
#         elif atom_type == 'all':
#             counter = counter_Lu + counter_Fe

#         fig, ax = plt.subplots(
#             nrows=counter, 
#             ncols=1, 
#             figsize=(6, counter*1.5),
#             sharex=True, 
#             sharey=True
#         )

#         counter = 0
#         counter_Lu = 0
#         counter_Fe = 0
#         for window_ind in range(len(self.layers)):
#             if type == 'mean':
#                 intensities = self.layers[window_ind].intensities.values()
#             elif type == 'max':
#                 intensities = self.layers[window_ind].max_intensities.values()
                
#             if atom_type == 'all':
#                 if self.layers[window_ind].atom_type == 'Lu':
#                     ax[counter].scatter(self.layers[window_ind].atom_positions[:, 1], list(intensities), s = 5, color='r', alpha=0.7, label = 'Lu')
#                 elif self.layers[window_ind].atom_type == 'Fe':
#                     ax[counter].scatter(self.layers[window_ind].atom_positions[:, 1], list(intensities), s = 5, color='b', alpha=0.7, label = 'Fe')
#             else:
#                 if self.layers[window_ind].atom_type == 'Lu':
#                     ax[counter].scatter(self.layers[window_ind].atom_positions[:, 1], list(intensities), s = 5, color='r', alpha=0.7)
#                 elif self.layers[window_ind].atom_type == 'Fe':
#                     ax[counter].scatter(self.layers[window_ind].atom_positions[:, 1], list(intensities), s = 5, color='b', alpha=0.7)
#             counter += 1

#         ax[0].set_title("Scatters Stacked")
#         plt.tight_layout()
        
#     def plot_intensity_KDE_stacked(self, atom_type = 'all', bins=20, type = 'mean'):
#         """
#         Plot the stacked histogram with GMM of intensities for all layers in the crop.

#         atom_type: str
#             The type of atoms to consider for intensity calculation.
#             ('all', 'Fe', 'Lu')
#             for 'all', it will plot histogram for both atoms sharing the same x-axis.
#         """

#         intensities_dict = {}
#         mean_dict = {}
#         std_dict = {}

#         # Record all histogram info: intensities, mean, std
#         counter = 0
#         for window_ind in range(len(self.layers)):
#             if type == 'mean':
#                 values = self.layers[window_ind].intensities.values()
#                 mean = self.layers[window_ind].mean_intensities
#                 std = self.layers[window_ind].std_intensities
#             elif type == 'max':
#                 values = self.layers[window_ind].max_intensities.values()
#                 mean = self.layers[window_ind].mean_max_intensities
#                 std = self.layers[window_ind].std_max_intensities

#             if self.layers[window_ind].atom_type == atom_type or atom_type == 'all':
#                 intensities_dict[counter] = values
#                 mean_dict[counter] = mean
#                 std_dict[counter] = std
#                 counter += 1
        
#         # Plot creation
#         fig, ax = plt.subplots(
#             nrows=counter, 
#             ncols=1, 
#             figsize=(10, counter),
#             sharex=True, 
#             sharey=True
#         )
#         if counter == 1:
#             ax = [ax]

#         counter = 0
#         for i in range(len(self.layers)):
#             # Plot histogram of intensities 
#             if self.layers[i].atom_type == 'Lu':
#                 ax[counter].hist(intensities_dict[i], bins=bins, density = True, alpha=0.2, label='Lu', color='r')
#                 mean = mean_dict[i]
#                 std = std_dict[i]
#                 ax[counter].axvline(mean, color='black', linestyle='dashed', linewidth=2, label='Mean')
#                 ax[counter].axvspan(mean - std, mean + std, alpha=0.2, color='g', label='std')
#             elif self.layers[i].atom_type == 'Fe':
#                 ax[counter].hist(intensities_dict[i], bins=bins, density = True, alpha=0.2, label='Fe', color='b')           
#                 mean = mean_dict[i]
#                 std = std_dict[i]
#                 ax[counter].axvline(mean, color='black', linestyle='dashed', linewidth=2, label='Mean')
#                 ax[counter].axvspan(mean - std, mean + std, alpha=0.2, color='g', label='std')

#             arr = list(intensities_dict[i])
#             self.layers[i].get_KDE_fit(type=type, get_plot=False)
#             x = np.linspace(min(arr)-0.2, max(arr)+0.2, 500)
#             if self.layers[i].atom_type == 'Lu':
#                 ax[counter].plot(x, self.layers[i].kde(x), "-r", label="KDE fit")
#             elif self.layers[i].atom_type == 'Fe':
#                 ax[counter].plot(x, self.layers[i].kde(x), "-b", label="KDE fit")

#             counter += 1

#         if type == 'mean':
#             ax[0].set_title('Stacked Histogram of Mean Intensities and KDE Fit')
#         elif type == 'max':
#             ax[0].set_title('Stacked Histogram of Max Intensities and KDE Fit')
#         fig.subplots_adjust(hspace=0)
#         plt.tight_layout()
