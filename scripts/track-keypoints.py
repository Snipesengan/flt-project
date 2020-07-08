#!/usr/bin/env python3

import argparse
import json

import cv2
import numpy as np


MIN_MATCH_COUNT = 10


def create_flann():
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    return cv2.FlannBasedMatcher(index_params, search_params)


def draw_matches(img1, kp1, img2, kp2, good, matchesMask):
    draw_params = dict(
                matchColor=(0, 255, 0),  # draw matches in green color
                singlePointColor=None,
                matchesMask=matchesMask,  # draw only inliers
                flags=2
    )
    img = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    return img


def _main(args):

    # load detector
    with open(args.config_file, "r") as fp:
        config = json.load(fp)
        detector = cv2.xfeatures2d.SIFT_create(**config['SIFT'])

    # initialise keypoints matcher
    # bf = cv2.BFMatcher(normtype=[_args.normtype], crosscheck=False)
    flann = create_flann()

    # detect keypoints of the first frame
    img1 = cv2.imread(args.target[0], 0)
    h, w = img1.shape
    kp1, des1 = detector.detectAndCompute(img1, None)

    # Sliding window of two frames
    for img_path in args.target[1:]:
        img2 = cv2.imread(img_path, 0)

        # detect and match
        kp2, des2 = detector.detectAndCompute(img2, None)
        matches = flann.knnMatch(des1, des2, k=2)  # k=2 for lowe ratio

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < args.lowe_ratio * n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            # sample keypoints only in good
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
            src_pts = src_pts.reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
            dst_pts = dst_pts.reshpae(-1, 2, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
            pts = pts.reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            img2 = cv2.polylines(
                img2,
                [np.int32(dst)],
                True, 255, 3, cv2.LINE_AA
            )

        else:
            print("Not enough matches are found - %d/%d" % (
                len(good), MIN_MATCH_COUNT))
            matchesMask = None

        img3 = draw_matches(img1, kp1, img2, kp2, good, matchesMask)
        cv2.imshow('gray', img3)
        cv2.waitKey(0)
        img1 = img2


def _get_parser():
    parser = argparse.ArgumentParser(
        description="CLI to track keypoints between a series of images"
    )
    parser.add_argument("target", type=str, nargs="+", help="path to image/s")
    parser.add_argument("-cfg", "--config_file", type=str, nargs="?",
                        default="config.json",
                        help="path to configuration file")
    parser.add_argument("--lowe_ratio", type=float, default=0.7, nargs='?',
                        help="lowe's ratio for filtering matches")
    return parser


if __name__ == '__main__':
    parser = _get_parser()
    try:
        _main(parser.parse_args())
    except KeyboardInterrupt:
        print("program terminated by user")

    cv2.destroyAllWindows()
    print("exiting..")
