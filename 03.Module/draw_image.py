##################################################
################# Import Modules #################
##################################################
import numpy as np
import cv2 as cv
##################################################
#################### Utilities ###################
##################################################
### Local  Utilities (sub)

### Local  Utilities (global)
### 1. Generate an image with mask on the "image_origin"
'''
image_with_mask = generated_image_with_a_mask(image_origin, mask_prediction, alpha = 0.6)
'''
def generated_image_with_a_mask(image_origin, mask_prediction, alpha = 0.6):
    ### .1 Copy an image from the "image_origin"
    image_to_draw = image_origin.copy()
    ### .2 Create "image_b"/"g"/"r" as masks from "mask_prediction"
    image_b = np.add((1-alpha)*image_to_draw[:, :, 0], alpha*255*np.ones_like(image_to_draw[:, :, 0]))
    image_g = np.add((1-alpha)*image_to_draw[:, :, 1], alpha*255*np.ones_like(image_to_draw[:, :, 1]))
    # image_r = np.add((1-alpha)*image_to_draw[:, :, 2], alpha*255*np.ones_like(image_to_draw[:, :, 2]))
    ### .3 Integrate "image_to_draw" with "image_b"/"g"/"r"
    image_to_draw[:, :, 0] = np.where(mask_prediction[:, :]!=0, image_b, image_to_draw[:, :, 0])
    image_to_draw[:, :, 1] = np.where(mask_prediction[:, :]!=0, image_g, image_to_draw[:, :, 1])
    # image_to_draw[:, :, 2] = np.where(mask_prediction[:, :]!=0, image_r, image_to_draw[:, :, 2])
    ### .4 Return the plotted image
    return image_to_draw
### 2. Generate an image with points on the "image_origin"
'''
image_with_points = generated_image_with_points(image_origin, points_coordinates, points_labels, marker_size=5)
'''
def generated_image_with_points(image_origin, points_coordinates, points_labels, marker_size=5):
    ### .1 Copy an image from the "image_origin"
    image_to_draw = image_origin.copy()
    ### .2 Extract coordinates for positive/negative points
    positive_points_coordinates = points_coordinates[points_labels==1]
    negative_points_coordinates = points_coordinates[points_labels==0]
    ### .3 Plot points on the "image_origin"
    ###    Green points for positive prompt
    for coordinate in positive_points_coordinates:
        cv.circle(image_to_draw, coordinate, 0, (  0,255,  0), marker_size)
    ###    Red   points for negative prompt
    for coordinate in negative_points_coordinates:
        cv.circle(image_to_draw, coordinate, 0, (  0,  0,255), marker_size)
    ### .4 Return the plotted image
    return image_to_draw
### 3. Generate an image with a bounding box on the "image_origin"
'''
image_with_box = generated_image_with_a_box(image_origin, box_prediction, color=(  0,255,  0), thickness=2)
'''
def generated_image_with_a_box(image_origin, box_prediction, color=(  0,255,  0), thickness=2):
    ### .1 Copy an image from the "image_origin"
    image_to_draw = image_origin.copy()
    ### .2 Extract informations from the "box_prediction"
    x_center, y_center = box_prediction[0], box_prediction[1]
    width   , height   = box_prediction[2] - box_prediction[0], box_prediction[3] - box_prediction[1]
    ### .3 Calculate coordinates of 4 vertices to "box_prediction"
    coordinate_top_left     = (x_center-width, y_center-height)
    coordinate_top_right    = (x_center+width, y_center-height)
    coordinate_bottom_left  = (x_center-width, y_center+height)
    coordinate_bottom_right = (x_center+width, y_center+height)
    ### .4 Plot the bounding box on the image
    cv.line(image_to_draw, coordinate_top_left    , coordinate_top_right   , color, thickness)
    cv.line(image_to_draw, coordinate_top_right   , coordinate_bottom_right, color, thickness)
    cv.line(image_to_draw, coordinate_bottom_right, coordinate_bottom_left , color, thickness)
    cv.line(image_to_draw, coordinate_bottom_left , coordinate_top_left    , color, thickness)
    ### .5 Return the plotted image
    return image_to_draw
##################################################
#################### Main Code ###################
##################################################
if __name__=='__main__': 1
    
    