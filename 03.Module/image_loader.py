##################################################
################# Import Modules #################
##################################################
import cv2 as cv
import numpy as np
from pathlib import Path
from os import listdir
##################################################
#################### Utilities ###################
##################################################
### Local  Utilities (sub)

### Local  Utilities (global)
### 1. Load all images within the directory with file names
'''
images_list, file_names_list = load_images_list_from_directory(path_to_directory='./01.Input Images/Test Images/')
'''
def load_images_list_from_directory(
        path_to_directory='./01.Input Images/Test Images/'):
    ### .1 Generate the "file_names_list" within the "path_to_directory"
    file_names_list = listdir(path_to_directory)
    ###    Filter the "directory_name" from "file_names_list"
    for file_name_index in range(len(file_names_list)-1,-1,-1):
        if Path(path_to_directory+file_names_list[file_name_index]).is_dir(): del file_names_list[file_name_index]
        if file_names_list[file_name_index][-4:] == '.csv': del file_names_list[file_name_index]
    ### .2 Load all images from "file_names_list"
    images_list = [cv.imread(path_to_directory+file_name, cv.IMREAD_UNCHANGED) for file_name in file_names_list]
    ### .3 Return the "images_list"
    return images_list, file_names_list

    ### .4 Test if the HSV differs the scenses
    from matplotlib import pyplot as plt
    file_index_list = []
    hsv_list, bgr_list = [], []
    for file_name_index in range(len(file_names_list)-1,-1,-1):
        file_index_list.append(int(file_names_list[file_name_index][:-4]))
        image = images_list[file_name_index]
        image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        hsv = np.average(image_hsv, axis=(0,1))
        bgr = np.average(image    , axis=(0,1))
        hsv_list.append(hsv)
        bgr_list.append(bgr)
    else: 
        hsv_array = np.array(hsv_list)
        print(hsv_array.shape)
    plt.scatter(file_index_list, hsv_array[:,0], c='r', s=50)
    plt.scatter(file_index_list, hsv_array[:,1], c='g', s=50)
    plt.scatter(file_index_list, hsv_array[:,2], c='b', s=50)
    plt.show()
    exit()
##################################################
#################### Main Code ###################
##################################################
if __name__=='__main__': 1
    
    