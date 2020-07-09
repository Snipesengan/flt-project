import os

import cv2
import numpy as np

from flt.exceptions import (CalibrationFolderExistsError,
                            ChessboardNotFoundError,
                            InconsistentCalibrationSizeError)


class StereoManager(object):

    """
    Manages multiple pairs of stereo cameras.

    """

    def __init__(self):
        pass


class StereoCalibration(object):

    """
    Stereo camera calibration for pinhole model cameras.

    P = [R|t] * k

    P := camera matrix, R := rotation matrix, t := translation vector,
    k := intrinsic matrix

    This object stores the calibration paramters for a stereo camera.
    It has methods to do basic 3D reconstruction.

    """

    def _save_dir(self, outdir, override=False):
        if os.path.isdir(outdir) and not override:
            raise CalibrationFolderExistsError

        for k, v in self.__dict__.items():
            outname = '{}.{}'.format(k, 'npy')
            np.save(os.path.join(outdir, outname), v, allow_pickle=False)

    def __init__(self):
        self.k_mat = [None, None]           # Intrinsic camera matrix
        self.dist_coefs = [None, None]      # Distortion coefficients
        self.R_mat = None                   # Camera rotation matrix
        self.t_vec = None                   # Translation vector
        self.e_mat = None                   # Essential matrix
        self.f_mat = None                   # Fundamental matrix

    def save_dir(self, outdir, override=False):
        self._save_dir(outdir, override)


class StereoCalibrator(object):

    def _find_corners(self, image):
        """ Find calibration corners in an image containing pattern. """

        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH
        chessboard_flags += cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(image,
                                                 (self.rows, self.cols),
                                                 flags=chessboard_flags)
        if not ret:  # no patterns was found
            return

        flags = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.01)
        cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), flags)

        return corners

    def _show_corners(self, image, corners):
        """Show chessboard corners found in image."""
        tmp = image.copy()
        cv2.drawChessboardCorners(tmp, (self.rows, self.cols), corners, True)

        window_name = "Chessboard"
        cv2.imshow(window_name, tmp)
        if cv2.waitKey(0):
            cv2.destroyWindow(window_name)

    def __init__(self, rows, cols, square_size, image_size):

        self.n_img = 0  # number of calibration images
        self.rows = rows  # number of inside corners in chessboard's row
        self.cols = cols  # number of inside corners in chessboard's columns
        self.square_size = square_size  # size of chessboard squares in cm
        self.image_size = image_size  # size of images in pixel

        pattern_size = (self.rows, self.cols)
        # prepare object points
        objpt = np.zeros((np.prod(pattern_size), 3), np.float32)
        objpt[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        objpt *= self.square_size
        self.corner_coordinates = objpt
        self.object_points = []  # 3d points in real world space
        self.image_points = [[], []]  # 2d points in image plane

    def add_corners(self, image_pair, display=False):

        image1, image2 = image_pair
        correct_dims = image1.shape[-2::-1] == self.image_size and \
            image1.shape == image2.shape

        if not correct_dims:
            raise InconsistentCalibrationSizeError

        for i, image in enumerate(image_pair):

            corners = self._find_corners(image)
            if corners is None:
                if i == 1:  # remove first's corners
                    self.image_points[0].pop(-1)
                raise ChessboardNotFoundError

            if display:
                self._show_corners(image, corners)

            self.image_points[i].append(corners)
        self.object_points.append(self.corner_coordinates)

    def calibrate_cameras(self):
        """Calibrate camera based on found chessboard corners."""
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TermCriteria_EPS,
                    100, 1e-5)
        flags = (cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST +
                 cv2.CALIB_SAME_FOCAL_LENGTH)

        calib = StereoCalibration()

        _, calib.k_mat[0], calib.dist_coefs[0], calib.k_mat[1], \
            calib.dist_coefs[1], calib.R_mat, calib.t_vec, calib.e_mat, \
            calib.f_mat, per_view_error = cv2.stereoCalibrateExtended(
                self.object_points, self.image_points[0],
                self.image_points[1], calib.k_mat[0], calib.dist_coefs[0],
                calib.k_mat[1], calib.dist_coefs[1], self.image_size,
                calib.R_mat, calib.t_vec, flags=flags, criteria=criteria)

        return calib

    def get_num_images(self):
        """ return total number of images that were found for calibration """
