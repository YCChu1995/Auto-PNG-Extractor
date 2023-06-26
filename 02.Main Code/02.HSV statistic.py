##################################################
################# Import Modules #################
##################################################
import numpy as np
from csv import writer
from ultralytics import YOLO
import cv2 as cv
from matplotlib import use
from matplotlib import pyplot as plt
use('TkAgg')
from sys import path
path += ['./03.Module']
from image_loader import load_images_list_from_directory
from sam_model_initialization import mask_predictor_initialization
##################################################
#################### Utilities ###################
##################################################
### Local Utilities (sub)
###    Convert RGB to HSV
'''
hsv = ____bgr_to_hsv(bgr_value)
'''
def ____bgr_to_hsv(bgr_value):
    ### .1 Preprocess the BGR value
    b, g, r = bgr_value/255
    color_max, color_min = max((b, g, r)), min((b, g, r))
    color_dif = color_max - color_min
    ### .2 Calculate the HSV value
    ###    Hue
    if color_dif < 0.001: 
        h = 0
    else:
        if   color_max == r:
            if g >= b:
                h = 60*(g-b)/color_dif
            else:
                h = 60*(g-b)/color_dif + 360
        elif color_max == g:
            h = 60*(b-r)/color_dif + 120
        elif color_max == b:
            h = 60*(b-g)/color_dif + 240
    ###    Saturation
    if color_max == 0:
        s = 0
    else:
        s = 1-(color_min/color_max)
    ###    Value
    v = color_max
    ### .3 Return 
    return (h, s, v)

### Local Utilities (main)
def main():
    from pathlib import Path
    from os import listdir
    path_to_directory = './01.Intput Images/Test Images/'
    ### .1 Generate the "file_name_list" within the "path_to_directory"
    file_name_list = listdir(path_to_directory)
    ###    Filter the "directory_name" from "file_name_list"
    for file_name_index in range(len(file_name_list)-1,-1,-1):
        if Path(path_to_directory+file_name_list[file_name_index]).is_dir(): del file_name_list[file_name_index]
    
    ID_list = []
    h_list  = []
    s_list  = []
    v_list  = []
    for file_name_index in range(len(file_name_list)):
        image = cv.imread(path_to_directory+file_name_list[file_name_index])
        ID_list.append(int(file_name_list[file_name_index][:-4]))
        
        # bgr = np.average(image, axis=(0,1))
        hsv = ____bgr_to_hsv( np.average(image, axis=(0,1)) )

        h_list.append(hsv[0])
        s_list.append(hsv[1])
        v_list.append(hsv[2])
    
    ### .4 Test if the HSV differs the scenses
    from matplotlib import pyplot as plt
    # plt.scatter(ID_list, h_list, c='r', s=50)
    plt.scatter(ID_list, s_list, c='g', s=50)
    plt.scatter(ID_list, v_list, c='b', s=50)
    plt.show()
##################################################
#################### Main Code ###################
##################################################
if __name__=='__main__':  main()