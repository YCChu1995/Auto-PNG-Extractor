##################################################
################# Import Modules #################
##################################################
from segment_anything import sam_model_registry
##################################################
#################### Utilities ###################
##################################################
### Local  Utilities (sub)

### Local  Utilities (global)
### 1. Initialize the "mask_generator"
'''
mask_generator = mask_generator_initialization(
                    sam_checkpoint = "./03.SAM Model/sam_vit_h_4b8939.pth",
                    model_type = "vit_h",
                    device = "cuda")
'''
def mask_generator_initialization(
        sam_checkpoint = "./03.SAM Model/sam_vit_h_4b8939.pth",
        model_type = "vit_h",
        device = "cuda"):
    ### .1 Initialize SAM
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    ### .2 Build the "mask_generator"
    from segment_anything import SamAutomaticMaskGenerator
    mask_generator = SamAutomaticMaskGenerator(sam)
    ### .3 Return the "mask_generator"
    return mask_generator
### 2. Initialize the ",mask_predictor"
'''
mask_predictor = mask_predictor_initialization(
                    sam_checkpoint = "./03.SAM Model/sam_vit_h_4b8939.pth",
                    model_type = "vit_h",
                    device = "cuda")
'''
def mask_predictor_initialization(
        sam_checkpoint = "./03.SAM Model/sam_vit_h_4b8939.pth",
        model_type = "vit_h",
        device = "cuda"):
    ### .1 Initialize SAM
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    ### .2 Build the "mask_predictor"
    from segment_anything import SamPredictor
    mask_predictor = SamPredictor(sam)
    ### .3 Return the "mask_predictor"
    return mask_predictor
##################################################
#################### Main Code ###################
##################################################
if __name__=='__main__': 1
    
    