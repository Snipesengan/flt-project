#!/usr/bin/env python3

import os
import time
import argparse

import cv2
from tqdm import tqdm

from flt.calibration import StereoCalibrator
from flt.exceptions import ChessboardNotFoundError
from flt.utils import split_views


def _main(args):

    calib = StereoCalibrator(args.pattern_rows, args.pattern_columns,
                             args.square_size, tuple(args.size))

    if not os.path.isdir(args.out_dir):
        print("Invalid output directory")
        return

    print("Detecting calibration pattern in images...")
    for img_path in tqdm(args.calibration_images):

        img = cv2.imread(img_path)
        left, center, right = split_views(img)
        try:
            if args.calib_side == 'left':
                image_pair = (left, center)
            elif args.calib_side == 'right':
                image_pair = (right, center)

            calib.add_corners(image_pair, args.display_corners)

        except ChessboardNotFoundError:
            print("No chessboard pattern found in {}".format(img_path))

    print("Calibrating stereo pair...")
    t1 = time.time()
    calibration = calib.calibrate_cameras()
    t2 = time.time()
    print("Time taken = {}s".format(t2 - t1))
    calibration.save_dir(args.out_dir, True)


def _get_parser():

    parser = argparse.ArgumentParser(
                        description="""CLI for calibrating stereo camera""")
    parser.add_argument("calibration_images", type=str, nargs="+",
                        help="path to calibration image/s")
    parser.add_argument("-o", "--out_dir", type=str, nargs="?",
                        default="/tmp",
                        help="output directory for calibration paramters")
    parser.add_argument("-r", "--pattern_rows", type=int, nargs="?",
                        default=7,
                        help="number of rows on chessboard pattern")
    parser.add_argument("-c", "--pattern_columns", type=int, nargs="?",
                        default=10,
                        help="number of columns on chessboard pattern")
    parser.add_argument("-sq", "--square_size", type=float, nargs="?",
                        default=2,
                        help="Square size in (cm)")
    parser.add_argument("-sz", "--size", type=int, nargs=2,
                        default=None,
                        required=True,
                        help="image size [width height] in pixel")
    parser.add_argument("--calib_side", type=str, nargs="?",
                        default='left',
                        choices=['left', 'right'],
                        help="which side to calibrate")
    parser.add_argument("--display_corners", action='store_true',
                        help="display detected corners in images")
    return parser


if __name__ == '__main__':
    parser = _get_parser()
    try:
        _main(parser.parse_args())
    except KeyboardInterrupt:
        print("Program terminated by user")

    cv2.destroyAllWindows()
