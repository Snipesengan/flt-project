#!/usr/bin/env python3

import os
import argparse

import cv2
from tqdm import tqdm

from flt.utils import split_views, mask_boxes, draw_boxes


def filter_largest_rectangle(rects):
    """Return the largest rectangle by area

    Args:
        rects (Tuple): x1, y1, x2, y2
    """
    calc_area = lambda x: (x[2] - x[0]) * (x[3] - x[1])
    out = sorted(rects, key=calc_area)
    return out[0]
    

def _main(args):
    if not os.path.isdir(args.save):
        raise OSError("save directory not found - {}".format(args.save))

    front_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    for target in tqdm(args.target):
        # read
        image = cv2.imread(target, 0)
        left, center, right = split_views(image)

        # detect & mask
        faces = front_cascade.detectMultiScale(left)
        face = filter_largest_rectangle(faces) 
        res = mask_boxes(left, [face], invert=True)

        # save
        _, name = os.path.split(target)
        cv2.imwrite(os.path.join(args.save,name), res)


def _get_parser():
    parser = argparse.ArgumentParser(
        description="Runs facial detection using Haar-Cascade")
    parser.add_argument("target", type=str, nargs="+",help="path to image/s")
    parser.add_argument("-s", "--save", type=str, default="/tmp",nargs=1,
        help="output directory")
    return parser


if __name__ == '__main__':
    parser = _get_parser()
    try:
        _main(parser.parse_args())
    except KeyboardInterrupt:
        print("Program terminated by user")

    cv2.destroyAllWindows()
        