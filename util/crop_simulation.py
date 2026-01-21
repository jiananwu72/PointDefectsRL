import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

import sys
from pathlib import Path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

class CropSimulation:
    def get_vertical_lines(self, get_plot=False, 
                        detect_width=1, min_sep=3, prom_coeff = 0.1):
        """
        Detect vertical peaks and troughs in the ROI by summing pixel intensities along vertical lines.

        Args:
            get_plot: If True, will plot the vertical lines and their intensities.
            detect_width: The width of the strip to sum over when calculating strengths.
            min_sep: Minimum separation (in pixels) between peaks/troughs.
            prom_coeff: Coefficient to determine the prominence threshold for peak detection.
        """

        if self.vertical_peaks is None:
            self.get_vertical_peaks(detect_width=detect_width, min_sep=min_sep, prom_coeff=prom_coeff)
        
        vertical_lines_x_coords = self.vertical_peaks
        vertical_lines = []
        for i in vertical_lines_x_coords:
            vertical_line = np.asarray(self.signal.isig[i, self.start:self.end])
            vertical_lines.append(vertical_line)

        if get_plot:
            n_lines = len(vertical_lines)
            y_axis_coords = np.arange(self.start, self.end)
            if n_lines > 10:
                print("Too many vertical lines, plotting the first 10 only.")
                n_lines = 10

            if n_lines > 0:
                fig, axes = plt.subplots(1, n_lines, sharey=True, figsize=(n_lines, 8))
                if n_lines == 1:
                    axes = [axes]
                for ax, line, x_coord in zip(axes, vertical_lines[:n_lines], vertical_lines_x_coords[:n_lines]):
                    ax.plot(line, y_axis_coords, '-k')
                    ax.set_title(f'x={x_coord}')
                    ax.set_xlabel('Int.')
                    ax.grid(True, alpha=0.3)
                axes[0].invert_yaxis()
                axes[0].set_ylabel('Y-axis pixel index')
                plt.subplots_adjust(wspace=0.1)
                plt.tight_layout()
                plt.show()
            else:
                print("No vertical lines detected to plot.")

        self.vertical_lines = vertical_lines
        self.vertical_lines_x_coords = vertical_lines_x_coords
        return vertical_lines, vertical_lines_x_coords

    def gaussian_filtering(self, get_plot=False, sigma=5):
        """
        Apply Gaussian filtering to the vertical lines for better peak detection.

        Args:
            get_plot: If True, will plot the filtered vertical lines.
            sigma: Standard deviation for Gaussian kernel.
        """

        filtered_vertical_lines = [gaussian_filter1d(line, sigma=sigma) for line in self.vertical_lines]

        if get_plot:
            n_lines = len(filtered_vertical_lines)
            y_axis_coords = np.arange(self.start, self.end)
            if n_lines > 10:
                print("Too many vertical lines, plotting the first 10 only.")
                n_lines = 10

            if n_lines > 0:
                fig, axes = plt.subplots(1, n_lines, sharey=True, figsize=(n_lines, 8))
                if n_lines == 1:
                    axes = [axes]
                for ax, line, x_coord in zip(axes, filtered_vertical_lines[:n_lines], self.vertical_lines_x_coords[:n_lines]):
                    ax.plot(line, y_axis_coords, '-k')
                    ax.set_title(f'x={x_coord}')
                    ax.set_xlabel('Int.')
                    ax.grid(True, alpha=0.3)
                axes[0].invert_yaxis()
                axes[0].set_ylabel('Y-axis pixel index')
                plt.subplots_adjust(wspace=0.1)
                plt.tight_layout()
                plt.show()
            else:
                print("No vertical lines detected to plot.")

        self.filtered_vertical_lines = filtered_vertical_lines
        return filtered_vertical_lines

    def get_vertical_line_peaks(self, peaks_min_distance=10):
        """
        Identify the peaks of vertical lines from their intensities.

        Args:
            peaks_min_distance: Minimum distance between peaks to consider local peak identification.
        """
        
        vertical_line_peaks = []
        if not hasattr(self, 'filtered_vertical_lines') or self.filtered_vertical_lines is None:
            vertical_lines = self.vertical_lines
        else:
            vertical_lines = self.filtered_vertical_lines

        for _, line in enumerate(vertical_lines):
            peaks, _ = find_peaks(line, distance=peaks_min_distance)
            vertical_line_peaks.append(peaks)
        
        self.vertical_line_peaks = vertical_line_peaks
        return vertical_line_peaks
    
    
        