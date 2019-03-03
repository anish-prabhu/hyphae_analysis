import math
import os
from os.path import isfile
from os.path import join

import cv2
import matplotlib as mpl
mpl.use('TkAgg')
import numpy as np
from matplotlib import pyplot as plt

_DEFAULT_DATA_DIR = './data/'
_DEFAULT_SAVE_DIR = './output/'
_DEFAULT_CROP_POLICY = False
_INTENSITY_THRESHOLD = 128
_MIN_CIRCULARITY, _MAX_CIRCULARITY = 0.8, 1.2
_MIN_PERIMETER = 50


def process_data(data_dir=_DEFAULT_DATA_DIR,
                 save_dir=_DEFAULT_SAVE_DIR,
                 crop=_DEFAULT_CROP_POLICY,
                 threshold=None):
    # Create save directory
    saved_image_dir = os.path.join(save_dir, 'images')
    os.makedirs(saved_image_dir, exist_ok=True)

    # Retrieve all images in the data directory
    print(data_dir)

    subdirs = [
        x for x in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, x))
    ]
    print(subdirs)
    for subdir in subdirs:
        subdir_path = os.path.join(data_dir, subdir)
        fnames = [
            f for f in os.listdir(subdir_path) if isfile(join(subdir_path, f))
        ]
        fnames.sort()
        areas, output_fnames = [], []
        print(fnames)
        for fname in fnames:
            # Retrieve the image
            img_path = subdir_path + '/' + fname
            print(img_path)
            image = cv2.imread(str(img_path), 1)
            # normalize image
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

            # Ensure a valid image was read in
            if image is None:
                continue

            print("Processing {}...".format(fname))
            # Detect Microbial Structure
            area, feeding_structures = detect_hyphae_area(
                image, crop=crop, threshold=threshold)
            areas.append(area)
            output_fnames.append(fname)
            output_image_path = os.path.join(saved_image_dir,
                                             _sanitize_name(str(fname)))
            cv2.imwrite(output_image_path, feeding_structures)

        # Write output with format easy to copy paste
        output_path = os.path.join(save_dir, _sanitize_name(subdir) + ".txt")
        with open(output_path, "a") as f:
            f.write("File Names:\n\n")
            for fname in output_fnames:
                f.write("{}\n".format(fname))

            f.write("\nFeeding Strucure Areas:\n\n")
            for area in areas:
                f.write("{}\n".format(area))


def detect_hyphae_area(img, crop=False, threshold=None):
    if crop:
        crop_mask = crop_img(img)
        if crop_mask is not None:
            cv_plot(img, "Cropped Image.")
    h, w = img.shape[:2]
    original_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST,
                                                cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [
        contour for contour in contours if valid_contour(contour, thresh)
    ]
    print("Kept {} / {} contours.".format(
        len(filtered_contours), len(contours)))
    feeding_structures = img.copy()
    mask = np.zeros(thresh.shape, np.uint8)
    cv2.drawContours(mask, filtered_contours, -1, 255, -1)
    cv2.drawContours(feeding_structures, filtered_contours, -1, (0, 255, 0), 1)
    if crop and crop_mask is not None:
        feeding_structures[crop_mask] = original_img[crop_mask]
    else:
        _zero_border(feeding_structures)
        _zero_border(mask)

    area = 1 - (np.count_nonzero(mask) / mask.size)
    display_img = np.hstack([original_img, feeding_structures])
    cv_plot(display_img, "Image with contours")
    return area, display_img


def _zero_border(img):
    img[0, :] = 0
    img[-1, :] = 0
    img[:, 0] = 0
    img[:, -1] = 0
    return img


def valid_contour(cnt, thresh):
    perimeter = cv2.arcLength(cnt, True)
    if perimeter < _MIN_PERIMETER:
        return False

    area = cv2.contourArea(cnt)
    circularity = 4 * math.pi * (area / (perimeter * perimeter))
    mask = np.zeros(thresh.shape, np.uint8)
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(mask, center, radius, 255, cv2.FILLED)
    pixelpoints = np.nonzero(mask)
    pixel_intensities = thresh[pixelpoints].flatten()
    pixel_intensities.sort()
    pixel_intensities = pixel_intensities[::-1]
    median_intensity = pixel_intensities[int(len(pixel_intensities) * 0.50)]
    mean_intensity = pixel_intensities.mean()
    if median_intensity < _INTENSITY_THRESHOLD:
        return False
    return True


def crop_img(img):
    h, w = img.shape[:2]
    pts = get_pts(img)
    plt.close('all')
    if len(pts) < 1:
        return None
    mask = np.zeros((h, w))
    cv2.fillPoly(mask, [pts], color=1)
    cords = np.where(mask != 1)
    img[cords] = 0
    return cords


def get_pts(img, tout=-1, bgr=True):
    # Set ginput to retrieve clicks
    if bgr:
        img = img[..., ::-1]

    plt.imshow(img, cmap='gray')
    pts = np.array(plt.ginput(0, timeout=tout)).astype(int)
    plt.show()
    return pts


def cv_plot(img, name, disp_time=1000, window_height=500, window_width=500):
    # Create cv2 window
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, window_height, window_width)

    # Show the image
    cv2.imshow(name, img)
    cv2.waitKey(disp_time)
    cv2.destroyAllWindows()


_ACCEPTABLE_NON_ALPHANUM = ('.', '_', '-', '/')
def _sanitize_name(name):
    return "".join([
        c for c in name
        if c.isalpha() or c.isdigit() or c in _ACCEPTABLE_NON_ALPHANUM
    ]).rstrip()


if __name__ == "__main__":
    process_data()

