##################################################
################# Import Modules #################
##################################################
import numpy as np
from csv import writer
from ultralytics import YOLO
import cv2 as cv
from matplotlib import use
from matplotlib import pyplot as plt
# use('TkAgg')
from sys import path
path += ['./02.Module']
from image_loader import load_images_list_from_directory
from sam_model_initialization import mask_predictor_initialization
##################################################
#################### Utilities ###################
##################################################
### Local Utilities (sub)
### 1. Load models for encoder and decoder
'''
model_detection, mask_predictor = __load_models()
'''
def __load_models():
    ### .1 Load the encoder, models from yolo
    ###    YOLO label: 0: 'person',    (100,100,255)R_light red
    ###                1: 'bicycle'    (255,100,255)R_pink
    ###                2: 'car',       (100,255,100)G_lime
    ###                3: 'motorcycle' (100,255,255)B_sky blue
    ###                4: 'airplane'
    ###                5: 'bus'        (100,255,100)G_lime
    ###                6: 'train'      (100,255,100)G_lime
    ###                7: 'truck'      (100,255,100)G_lime
    model_detection = YOLO("./03.Encoder Model/yolov8x.pt")
    # model_segmentation = YOLO("./03.Encoder Model/yolov8x_seg.pt")
    ### .2 Load the decoder, the "mask_predictor" from SAM 
    mask_predictor = mask_predictor_initialization(
                        sam_checkpoint = "./03.Decoder Model/sam_vit_h_4b8939.pth",
                        model_type = "vit_h",
                        device = "cuda")
    ### .3 Return models
    return model_detection, mask_predictor
### 2. Predict images with encoder in Batch
'''
encoder_results_list = __predict_encoder(images_list, model_detection)
'''
def __predict_encoder(images_list, model_detection):
    ### .1 Prepare parameters to cut batches
    length     = len(images_list)
    batch_size = 10
    ### .2 Prepare a list to storage 
    encoder_results_list = []
    ### .3 Predict the image batch by batch
    for batch_index in range(length//batch_size):
        encoder_results_list_current_batch = model_detection(images_list[batch_index*batch_size:(batch_index+1)*batch_size])
        encoder_results_list += [result.boxes.data.cpu().numpy()  for result in encoder_results_list_current_batch]
    else:
        encoder_results_list_current_batch = model_detection(images_list[(batch_index+1)*batch_size:])
        encoder_results_list += [result.boxes.data.cpu().numpy()  for result in encoder_results_list_current_batch]
    ### .4 Return results from the encoder
    return encoder_results_list
### 3. Show results from the encoder
'''
__show_encoder_results(images_list, encoder_results_list)
'''
def __show_encoder_results(images_list, encoder_results_list):
    ### .1 Keep displaying results
    image_index = 0
    while True:
        ### .2 Copy an image from the "images_list"
        image_to_draw = images_list[image_index].copy()
        ### .3 Decorate the "image_to_draw"
        ###    Note the title 
        cv.putText(image_to_draw, 'Image ID : '+str(image_index), (5,35),cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
        ###    Draw results from the encoder
        ____draw_bboxes(image_to_draw, encoder_results_list[image_index], font_scale=0.4)
        ### .4 Display the drawed image
        cv.imshow('', image_to_draw)
        ### .5 Let user determine which image to view
        key = cv.waitKey(1)
        if   key in (ord('s'), ord('S')): cv.destroyAllWindows(); break
        elif key in (ord('a'), ord('A')): image_index -= 1
        elif key in (ord('d'), ord('D')): image_index += 1
        ###    Safety check on the "image_index"
        if   image_index < 0: image_index = 0
        elif image_index >= len(images_list): image_index = len(images_list)-1
###    Show results from the decoder
'''
__show_decoder_results(images_list, encoder_results_list, decoder_mask_results_list)
'''
def __show_decoder_results(images_list, encoder_results_list, decoder_mask_results_list, decoder_bbox_results_list):
    ### .1 Keep displaying results
    image_index = 0
    while True:
        ### .2 Copy an image from the "images_list"
        image_to_draw = images_list[image_index].copy()
        ### .3 Decorate the "image_to_draw"
        ###    Note the title 
        cv.putText(image_to_draw, 'Image ID : '+str(image_index), (5,35),cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
        ###    Draw results from the encoder
        ____draw_bboxes(image_to_draw, encoder_results_list[image_index], font_scale=0.4)
        ###    Draw results from the decoder
        ____draw_masks (image_to_draw, decoder_mask_results_list[image_index], encoder_results_list[image_index], alpha=0.3)
        ###    Draw results from the decoder
        ____draw_bboxes_vanilla(image_to_draw, decoder_bbox_results_list[image_index])
        
        image_to_save_name = './05.Drawed Images/'+str(image_index)+'.jpg'
        cv.imwrite(image_to_save_name, image_to_draw)

        ### .4 Display the drawed image
        cv.imshow('', image_to_draw)
        ### .5 Let user determine which image to view
        key = cv.waitKey(1)
        if   key in (ord('s'), ord('S')): cv.destroyAllWindows(); break
        elif key in (ord('a'), ord('A')): image_index -= 1
        elif key in (ord('d'), ord('D')): image_index += 1
        ###    Safety check on the "image_index"
        if   image_index < 0: image_index = 0
        elif image_index >= len(images_list): image_index = len(images_list)-1
###    Draw bboxes on the "image_to_draw"
'''
____draw_bboxes(image_to_draw, detection_result, font_scale=0.2)
'''
def ____draw_bboxes(image_to_draw, encoder_result, font_scale=0.2):
    ### .1 Iterate through all "encoder_result"
    for encoder_result_index in range(encoder_result.shape[0]):
        ### .2 Filter the one with low confidence score
        if encoder_result[encoder_result_index,4] < 0.75: continue
        ### .3 Assign "color" and "text" from class and confidence score
        match encoder_result[encoder_result_index,5]:
            ###    YOLO label: 0: 'person',    (100,100,255)R_light red
            ###                1: 'bicycle'    (255,100,255)R_pink
            ###                2: 'car',       (100,255,100)G_lime
            ###                3: 'motorcycle' (100,255,255)B_sky blue
            ###                4: 'airplane'
            ###                5: 'bus'        (100,255,100)G_lime
            ###                6: 'train'
            ###                7: 'truck'      (100,255,100)G_lime
            case 0: color = (100,100,255); text = 'person'
            case 1: color = (255,100,255); text = 'bicycle'
            case 2: color = (100,255,100); text = 'car'
            case 3: color = (100,255,255); text = 'motorcycle'
            case 5: color = (100,255,100); text = 'bus'
            case 7: color = (100,255,100); text = 'truck'
            case _: continue
        ### .4 Draw the bbox
        cv.rectangle(image_to_draw, encoder_result[encoder_result_index,0:2].astype(np.uint16), encoder_result[encoder_result_index,2:4].astype(np.uint16), color, 1)
        ### .5 Put informations about the class and the confidence score
        cv.putText  (image_to_draw, text+' '+str(round(encoder_result[encoder_result_index,4],2)), (int(encoder_result[encoder_result_index,0]), int(encoder_result[encoder_result_index,1]-2)), cv.FONT_HERSHEY_SIMPLEX, font_scale, color)
###    Draw masks on the "image_to_draw"
'''
____draw_masks(image_to_draw, decoder_mask_result, alpha=0.6)
'''
def ____draw_masks(image_to_draw, decoder_mask_result, encoder_result, alpha=0.6):
    ### .1 Iterate through all "encoder_result"
    for decoder_result_index in range(len(decoder_mask_result)):
        ### .2 Assign "color" and "text" from class and confidence score
        match encoder_result[decoder_result_index,5]:
            ###    YOLO label: 0: 'person',    (100,100,255)R_light red
            ###                1: 'bicycle'    (255,100,255)R_pink
            ###                2: 'car',       (100,255,100)G_lime
            ###                3: 'motorcycle' (100,255,255)B_sky blue
            ###                4: 'airplane'
            ###                5: 'bus'        (100,255,100)G_lime
            ###                6: 'train'
            ###                7: 'truck'      (100,255,100)G_lime
            case 0: color = (alpha*100,alpha*100,alpha*255)
            case 1: color = (alpha*255,alpha*100,alpha*255)
            case 2: color = (alpha*100,alpha*255,alpha*100)
            case 3: color = (alpha*100,alpha*255,alpha*255)
            case 5: color = (alpha*100,alpha*255,alpha*100)
            case 7: color = (alpha*100,alpha*255,alpha*100)
            case _: continue
        ### .3 Draw the mask
        image_b = np.add((1-alpha)*image_to_draw[:, :, 0], color[0]*np.ones_like(image_to_draw[:, :, 0]))
        image_g = np.add((1-alpha)*image_to_draw[:, :, 1], color[1]*np.ones_like(image_to_draw[:, :, 1]))
        image_r = np.add((1-alpha)*image_to_draw[:, :, 2], color[2]*np.ones_like(image_to_draw[:, :, 2]))
        image_to_draw[:, :, 0] = np.where(decoder_mask_result[decoder_result_index][0, :, :]==True, image_b, image_to_draw[:, :, 0])
        image_to_draw[:, :, 1] = np.where(decoder_mask_result[decoder_result_index][0, :, :]==True, image_g, image_to_draw[:, :, 1])
        image_to_draw[:, :, 2] = np.where(decoder_mask_result[decoder_result_index][0, :, :]==True, image_r, image_to_draw[:, :, 2])
###    Draw bboxes on the "image_to_draw"
'''
____draw_bboxes_vanilla(image_to_draw, decoder_result)
'''
def ____draw_bboxes_vanilla(image_to_draw, decoder_result):
    ### .1 Iterate through all "encoder_result"
    for decoder_result_index in range(decoder_result.shape[0]):
        ### .4 Draw the bbox
        cv.rectangle(image_to_draw, decoder_result[decoder_result_index,0:2].astype(np.uint16), decoder_result[decoder_result_index,2:4].astype(np.uint16), (0,0,255), 1)
### 4. Predict mask from SAM
'''
decoder_mask_results_list = __mask_prediction(mask_predictor, images_list, encoder_results_list)
'''
def __mask_prediction(mask_predictor, images_list, encoder_results_list):
    ### .1 Prepare a list to storage results from decoder
    decoder_mask_results_list = []
    ### .2 Predict with decoder
    for image_index in range(len(images_list)):
        decoder_mask_results_list.append([])
        ### .3 Embedding the image
        mask_predictor.set_image(images_list[image_index])
        ### .4 Iterate through prediction results from encoder
        for encoder_result_index in range(encoder_results_list[image_index].shape[0]):
            ### .5 Filter the one with low confidence score
            if encoder_results_list[image_index][encoder_result_index,4] < 0.75: continue
            ### .6 Predict with prompts
            masks = mask_predictor.predict(box=encoder_results_list[image_index][encoder_result_index,:4].astype(np.uint16), multimask_output=False)[0]
            ### .7 Storage result from decoder
            decoder_mask_results_list[image_index].append(masks)
    ### .8 Return results from the decoder
    return decoder_mask_results_list
### 5. Extract bboxes, "decoder_bbox_results_list", from masks, "decoder_mask_results_list"
'''
decoder_bbox_results_list, count_detected_results = __bbox_extractor(decoder_mask_results_list)
'''
def __bbox_extractor(decoder_mask_results_list):
    ### .1 Count on detected results
    count_detected_results = 0
    ### .2 Prepare a list to storage extracted results
    decoder_bbox_results_list = []
    ### .2 Iterate throuth images
    for image_index in range(len(decoder_mask_results_list)):
        decoder_bbox_results_list.append([])
        ### .3 Iterate throuth detected results
        for detected_results_index in range(len(decoder_mask_results_list[image_index])):
            ### .4 Extract the information from the "decoder_mask_results_list"
            count_detected_results += 1
            mask_location = np.where(decoder_mask_results_list[image_index][detected_results_index][0]==True)
            x_top_left     = np.min(mask_location[1])
            y_top_left     = np.min(mask_location[0])
            x_bottom_right = np.max(mask_location[1])
            y_bottom_right = np.max(mask_location[0])
            decoder_bbox_results_list[image_index].append((x_top_left, y_top_left, x_bottom_right, y_bottom_right))
        ### .5 Convert storaged list to numpy.array
        decoder_bbox_results_list[image_index] = np.array(decoder_bbox_results_list[image_index], dtype=np.uint16)
    ### .6 Return extracted bboxes and the count of detected results from filtered encoder results
    return decoder_bbox_results_list, count_detected_results
### 6. Save mask results from decoder as png file
'''
__save_png_files(images_list, decoder_bbox_results_list, decoder_mask_results_list)
'''
def __save_png_files(images_list, decoder_bbox_results_list, decoder_mask_results_list):
    ### .1 Prepare the file_name of the image_to_save
    image_to_save_index  = 0
    folder_to_save_image = './05.Saved Images'
    hsv_of_image_list    = [['image_index', 'hsv']]
    ### .2 Iterate through all images
    for image_index in range(len(images_list)):
        ### .3 Calculate the HSV of the image
        bgr = np.average(images_list[image_index], axis=(0,1))
        hsv = ____bgr_to_hsv(bgr)
        ### .4 Iterate through mask results from the decoder
        for decoder_result_index in range(decoder_bbox_results_list[image_index].shape[0]):
            #!!    Filter the one with weird mask
            #!!    Maybe a classifier as a filter
            ### .5 Extract the image in the bbox
            x_left   = decoder_bbox_results_list[image_index][decoder_result_index][1]
            x_right  = decoder_bbox_results_list[image_index][decoder_result_index][3]
            y_top    = decoder_bbox_results_list[image_index][decoder_result_index][0]
            y_bottom = decoder_bbox_results_list[image_index][decoder_result_index][2]
            image_to_save = images_list[image_index][x_left:x_right,y_top:y_bottom,:]
            ### .6 Clean the background in the bbox by setting the alpha
            ###    Alpha value : Background => 0; Element => 255
            alpha_to_save = np.where(decoder_mask_results_list[image_index][decoder_result_index][0][x_left:x_right,y_top:y_bottom]==True, 255, 0)
            image_to_save = np.dstack([image_to_save, alpha_to_save])
            ### .7 Save the image into a png file
            # cv.imshow('', image_to_save)
            # if cv.waitKey(0): pass
            image_to_save_index += 1
            image_to_save_name = folder_to_save_image+'/'+str(image_to_save_index)+'.png'
            cv.imwrite(image_to_save_name, image_to_save)
            ### .8 Save the hsv of the enviroment
            hsv_of_image_list.append([image_to_save_index, hsv[0], hsv[1], hsv[2]])
    ### .9 Save the hsv into .csv file
    ____save_hsv_to_csv_file(hsv_of_image_list, folder_to_save_image)
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
###    Save the list of hsv values to saved images into a csv file
'''
____save_hsv_to_csv_file(hsv_of_image_list, folder_to_save_image)
''' 
def ____save_hsv_to_csv_file(hsv_of_image_list, folder_to_save_image):
    with open(folder_to_save_image+'//hsv.csv', 'w', encoding='UTF8') as csv_file:
        csv_writer = writer(csv_file)
        csv_writer.writerows(hsv_of_image_list)

### Local Utilities (main)
def main():
    ### .1 Load images within the directory
    images_list, _ = load_images_list_from_directory(path_to_directory='./04.Dataset/00.Test Images/')

    ### .2 Load models for encoder and decoder
    model_detection, mask_predictor = __load_models()

    ### .3 Predict with encoder
    encoder_results_list = __predict_encoder(images_list, model_detection)

    ### .4 Show results from the encoder
    __show_encoder_results(images_list, encoder_results_list)

    ### .5 Predict with decoder 
    decoder_mask_results_list = __mask_prediction(mask_predictor, images_list, encoder_results_list)
    ###    Derive the new bounding box label
    decoder_bbox_results_list, count_detected_results = __bbox_extractor(decoder_mask_results_list)
    
    ### .6 Show results from the decoder
    __show_decoder_results(images_list, encoder_results_list, decoder_mask_results_list, decoder_bbox_results_list)
    
    ### .7 Save mask results from decoder as png file
    __save_png_files(images_list, decoder_bbox_results_list, decoder_mask_results_list)
##################################################
#################### Main Code ###################
##################################################
if __name__=='__main__':  main()