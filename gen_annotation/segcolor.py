import cv2
import numpy as np
from PIL import Image
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon


def get_colors(img):
    # Converts to HSV color format and attempts to detect greys from HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Grey is in between 0 and 50 on saturation, so we want to get out of that.
    lower = np.array([0, 51, 50], np.uint8)
    upper = np.array([179, 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_and(img, img, mask=mask)
    return mask


def get_submasks(mask_raw):
    gray_mask = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_mask, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    submasks = {}
    mask = np.zeros_like(mask_raw)
    for idx in range(len(contours)):
        cv2.drawContours(mask, contours, idx, 255, -1)
        out = np.zeros_like(mask_raw)
        out[mask == 255] = img[mask == 255]






def old_get_submasks(mask):
    width, height = mask.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    submasks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask.getpixel((x, y))[:3]

            # If the pixel is not black...
            if pixel != (0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                submask = submasks.get(pixel_str)
                if submask is None:
                    # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    submasks[pixel_str] = Image.new('1', (width + 2, height + 2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                submasks[pixel_str].putpixel((x + 1, y + 1), 1)

    return submasks


def make_submask_annotations(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation