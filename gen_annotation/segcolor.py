import cv2
import numpy as np
from PIL import Image
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon


def get_colors(img):
    """
    Extracts regions of color and returns as a color mask
    :param img: Image with regions of color dim(W,H,3)
    :return: BGR mask of regions dim(W,H,3)
    """
    # Converts to HSV color format and attempts to detect greys from HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Grey is in between 0 and 50 on saturation, so we want to get out of that.
    lower = np.array([0, 51, 50], np.uint8)
    upper = np.array([179, 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_and(img, img, mask=mask)
    return mask


def get_mask_categorized(mask_raw):
    """
    Returns list of categorized submasks
    :param mask_raw: Image with regions of color dim(W,H,3)
    :return: Dictionary of binary submasks, keyed by RGB color
    """
    rgb_mask = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2RGB)
    pil_mask = Image.fromarray(rgb_mask.astype('uint8'), 'RGB')

    # Code sourced from:
    # https://www.immersivelimit.com/create-coco-annotations-from-scratch

    pil_mask_categorized = {}
    width, height = pil_mask.size
    print("Looping through image now!")

    # TODO: OPTIMIZE BY SWITCHING FROM PIL TO NUMPY

    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = pil_mask.getpixel((x, y))[:3]

            # If the pixel is not black...
            if pixel != (0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = pil_mask_categorized.get(pixel_str)
                if sub_mask is None:
                    # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    pil_mask_categorized[pixel_str] = Image.new('1', (width + 2, height + 2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                pil_mask_categorized[pixel_str].putpixel((x + 1, y + 1), 1)

    # Loop through, OPTIMIZE THIS IN THE FUTURE
    mask_categorized = {}
    for rgb_key in pil_mask_categorized:
        sub_image = pil_mask_categorized.get(rgb_key)
        mask_categorized[rgb_key] = np.asarray(sub_image)
        # sub_image.show()

    return mask_categorized


def make_submask_annotations(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    # Code sourced from:
    # https://www.immersivelimit.com/create-coco-annotations-from-scratch

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        # TODO: CHANGE THIS DEPENDING ON ACCURACY TO RESULTS
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

    # TODO: ADD PROPER ANNOTATIONS TO ALIGN WITH COCO STANDARDS
    # TODO: ADD SEGMENTATION SECTION FOR MULTIPLE OBJECTS IN AN IMAGE

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