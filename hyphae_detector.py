import logging
import math
import os
from os import path

import cv2
import matplotlib as mpl
import numpy as np
import yaml
from matplotlib import pyplot as plt

logging.basicConfig()
logging.root.setLevel(logging.INFO)

mpl.use('TkAgg')

_DEFAULT_CONFIG_PATH = './config.yaml'


def process_data():
    args = _process_config()
    general_args = args['general_options']
    analysis_args = args['analysis_options']

    # Create save directory
    # TODO(Anish): Probably change save_dir to output_dir
    save_dir = general_args['output_dir']
    saved_image_dir = path.join(save_dir, 'images')
    os.makedirs(saved_image_dir, exist_ok=True)

    # Retrieve all images in the data directory
    data_dir = general_args['data_dir']
    if not path.isdir(data_dir):
        raise ValueError("Data directory {} does not exist.")

    logging.info("Processing data located {}".format(data_dir))

    subdirs = [
        x for x in os.listdir(data_dir) if path.isdir(path.join(data_dir, x))
    ]
    for subdir in subdirs:
        logging.info("\tProcessing images in {}".format(subdir))
        subdir_path = path.join(data_dir, subdir)
        fnames = [
            f for f in os.listdir(subdir_path)
            if path.isfile(path.join(subdir_path, f))
        ]
        fnames.sort()
        # TODO: Figure out best format for this.
        areas, output_fnames = [], []
        for fname in fnames:
            logging.info("\t\tProcessing image {}".format(fname))
            # Retrieve the image
            img_path = path.join(subdir_path, fname)
            image = cv2.imread(img_path, 1)
            # normalize image
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

            # Ensure a valid image was read in
            if image is None:
                continue

            # Detect Microbial Structure
            # TODO(Anish): Unpack args and pass to detect_hyphae_area
            # from the analysis_args. Maybe also move crop to this
            # region.
            crop = analysis_args['crop']
            intensity_thresh = analysis_args['intensity_threshold']
            min_circularity = analysis_args['min_circularity']
            min_perimeter = analysis_args['min_perimeter']
            area, feeding_structures = detect_hyphae_area(
                image, save_dir, crop, intensity_thresh, min_circularity,
                min_perimeter)
            areas.append(area)
            output_fnames.append(img_path)
            output_image_path = path.join(saved_image_dir,
                                          _sanitize_name(img_path))
            cv2.imwrite(output_image_path, feeding_structures)

        # Write output with format easy to copy paste
        output_path = path.join(save_dir, _sanitize_name(subdir) + ".txt")
        with open(output_path, "a") as f:
            f.write("File Names:\n\n")
            for fname in output_fnames:
                f.write("{}\n".format(fname))

            f.write("\nFeeding Strucure Areas:\n\n")
            for area in areas:
                f.write("{}\n".format(area))


# TODO(Anish): remove this save_dir
def detect_hyphae_area(img, save_dir, crop, intensity_thresh, min_circularity,
                       min_perimeter):
    h, w = img.shape[:2]
    original_img = img.copy()
    if crop:
        crop_mask = crop_img(img)
        if crop_mask is not None:
            cropped_display_img = img.copy()
            cropped_display_img[crop_mask] = 0
            cv_plot(img, "Cropped Image.")

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST,
                                                cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [
        contour for contour in contours if valid_contour(
            contour, thresh, intensity_thresh, min_circularity, min_perimeter)
    ]
    logging.info("\t\t\tKept {} / {} contours.".format(
        len(filtered_contours), len(contours)))
    all_contour_img = img.copy()
    cv2.drawContours(all_contour_img, contours, -1, (0, 255, 0), 1)

    feeding_structures = img.copy()
    mask = np.zeros(thresh.shape, np.uint8)

    cv2.drawContours(mask, filtered_contours, -1, 255, -1)
    cv2.drawContours(feeding_structures, filtered_contours, -1, (0, 255, 0), 1)
    if crop and crop_mask is not None:
        mask[crop_mask] = 0
        feeding_structures[crop_mask] = original_img[crop_mask]

    _zero_border(feeding_structures)
    _zero_border(mask)

    cv_plot(feeding_structures, "Detected hyphae.")

    area = 1 - (np.count_nonzero(mask) / mask.size)
    display_img = np.hstack([original_img, feeding_structures])
    return area, display_img


def _zero_border(img):
    img[0, :] = 0
    img[-1, :] = 0
    img[:, 0] = 0
    img[:, -1] = 0
    return img


def valid_contour(cnt, thresh, intensity_thresh, min_circularity,
                  min_perimeter):
    perimeter = cv2.arcLength(cnt, True)
    if perimeter < min_perimeter:
        return False

    area = cv2.contourArea(cnt)
    circularity = 4 * math.pi * (area / (perimeter * perimeter))
    mask = np.zeros(thresh.shape, np.uint8)
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(mask, center, radius, 255, cv2.FILLED)
    pixelpoints = np.nonzero(mask)
    # TODO(Anish): Should this be the thresholded or
    # original image. Also, rename to more explicit
    # name.
    pixel_intensities = thresh[pixelpoints].flatten()
    pixel_intensities.sort()
    pixel_intensities = pixel_intensities[::-1]
    # TODO(Anish): maybe remove
    median_intensity = pixel_intensities[int(len(pixel_intensities) * 0.50)]
    mean_intensity = pixel_intensities.mean()
    if mean_intensity < intensity_thresh:
        return False

    if min_circularity < circularity:
        return False

    return True


def crop_img(img):
    h, w = img.shape[:2]
    pts = get_pts(img)
    # TODO(Anish): should we be closing here?
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


def _process_config():
    with open(_DEFAULT_CONFIG_PATH) as fp:
        args = yaml.load(fp, Loader=yaml.FullLoader)
    return args


if __name__ == "__main__":
    process_data()
