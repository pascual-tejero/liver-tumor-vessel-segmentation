"""

Calculation of nine common evaluation metrics based on segmentation such as overlap and distance
"""

import math

import numpy as np
import scipy.spatial as spatial
import scipy.ndimage.morphology as morphology


class Metric():
    
    def __init__(self, real_mask, pred_mask, voxel_spacing):
        """
        :param real_mask: gold standard
        :param pred_mask: prediction result
        :param voxel_spacing: the spacing of the body data
        """
        self.real_mask = real_mask
        self.pred_mask = pred_mask
        self.voxel_sapcing = voxel_spacing

        self.real_mask_surface_pts = self.get_surface(real_mask, voxel_spacing)
        self.pred_mask_surface_pts = self.get_surface(pred_mask, voxel_spacing)

        self.real2pred_nn = self.get_real2pred_nn()
        self.pred2real_nn = self.get_pred2real_nn()

    # The following three functions are useful for extracting bounds and calculating minimum distances
    def get_surface(self, mask, voxel_spacing):
        """
        :param mask: ndarray
        :param voxel_spacing: the spacing of the body data
        :return: the real coordinates (in mm) of the surface point from which the array is extracted
        """

        # The convolution kernel uses a three-dimensional 18-neighbourhood

        kernel = morphology.generate_binary_structure(3, 2)
        surface = morphology.binary_erosion(mask, kernel) ^ mask

        surface_pts = surface.nonzero()

        surface_pts = np.array(list(zip(surface_pts[0], surface_pts[1], surface_pts[2])))

        # (0.7808688879013062, 0.7808688879013062, 2.5) (88, 410, 512)
        # The read data spacing and shape do not correspond, so it needs to be reversed.
        return surface_pts * np.array(self.voxel_sapcing[::-1]).reshape(1, 3)

    def get_pred2real_nn(self):
        """
        :return: minimum distance from the predicted surface voxel to the gold standard surface voxel
        """

        tree = spatial.cKDTree(self.real_mask_surface_pts)
        nn, _ = tree.query(self.pred_mask_surface_pts)

        return nn

    def get_real2pred_nn(self):
        """
        :return: minimum distance from the gold standard surface voxel to the predicted resultant surface voxel
        """
        tree = spatial.cKDTree(self.pred_mask_surface_pts)
        nn, _ = tree.query(self.real_mask_surface_pts)

        return nn

    # The following six indicators are based on the degree of overlap
    def get_dice_coefficient(self):
        """
        :return: dice coefficient numerator of dice coefficient denominator of dice coefficient (the latter two are used to calculate dice_global)
        """
        intersection = (self.real_mask * self.pred_mask).sum()
        union = self.real_mask.sum() + self.pred_mask.sum()

        return 2 * intersection / union, 2 * intersection, union

    def get_jaccard_index(self):
        """
        :return: Deckard factor
        """
        intersection = (self.real_mask * self.pred_mask).sum()
        union = (self.real_mask | self.pred_mask).sum()

        return intersection / union

    def get_VOE(self):
        """
        :return: Volumetric Overlap Error
        """

        return 1 - self.get_jaccard_index()

    def get_RVD(self):
        """
        :return: Relative Volume Difference
        """

        return float(self.pred_mask.sum() - self.real_mask.sum()) / float(self.real_mask.sum())

    def get_FNR(self):
        """
        :return: False negative rate
        """
        fn = self.real_mask.sum() - (self.real_mask * self.pred_mask).sum()
        union = (self.real_mask | self.pred_mask).sum()

        return fn / union

    def get_FPR(self):
        """
        :return: False positive rate
        """
        fp = self.pred_mask.sum() - (self.real_mask * self.pred_mask).sum()
        union = (self.real_mask | self.pred_mask).sum()

        return fp / union

    # The following three indicators are based on distance
    def get_ASSD(self):
        """
        :return: Average Symmetric Surface Distance
        """
        return (self.pred2real_nn.sum() + self.real2pred_nn.sum()) / \
               (self.real_mask_surface_pts.shape[0] + self.pred_mask_surface_pts.shape[0])

    def get_RMSD(self):
        """
        :return: Root Mean Square symmetric Surface Distance
        """
        return math.sqrt((np.power(self.pred2real_nn, 2).sum() + np.power(self.real2pred_nn, 2).sum()) /
                         (self.real_mask_surface_pts.shape[0] + self.pred_mask_surface_pts.shape[0]))

    def get_MSD(self):
        """
        :return: Maximum Symmetric Surface Distance
        """
        return max(self.pred2real_nn.max(), self.real2pred_nn.max())
