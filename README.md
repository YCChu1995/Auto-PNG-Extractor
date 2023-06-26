# Auto-PNG-Extractor-SAM
> This APP automatically extracts PNG images of any object of interest from any normal image.
>
> 🚧 The code is still under organizing.

## 1. Main Function
  - ### Input :

    1. `Target Class` (ex: [`main(targetClass='vehicle')`](https://github.com/YCChu1995/Auto-PNG-Extractor/blob/main/02.Main%20Code/01.SAM%20Test.py?plain=1#L348) in the [01.SAM Test.py](https://github.com/YCChu1995/Auto-PNG-Extractor/blob/main/02.Main%20Code/01.SAM%20Test.py))

    2. `Image(s)` (ex: [01.Input Images](/01.Input%20Images/)/[Test Images](/01.Input%20Images/Test%20Images))

    <img src="/01.Input%20Images/Test%20Images/137.jpg" width="300" height="200"> <img src="/01.Input%20Images/Test%20Images/140.jpg" width="300" height="200">

  - ### Output : 

    `extracted PNG images` in [05.Saved Images](/05.Saved%20Images/)

    <img src="/06.Test%20Result/Extracted%20Result.PNG" height="200">

## 2. Quick Start (🚧 NOT finished yet)
  1. Download the whole project file
  2. Make sure the Python version `python -V` is greater than 3.10
  3. `pip install -r requirement.txt`
  4. `Run 01.SAM Test.py` in [02.Main Code](/02.Main%20Code) → You should get results in [3.Sub Functions](#3-sub-functions)

## 3. Sub Functions
  1. Build B.Box prompts for SAM (any pre-trained model should work, here I use YOLO v8)
     
     <img src="/06.Test%20Result/Encoder%20Result.png" height="200">

     - `Green B.Box` : Prediction results from the pre-trained model with an overcritical threshold
       
     - Explanation - Why did I choose an **overcritical** threshold ?
       
       1. "False Positive" in the confusion matrix is **unacceptable**, cause it will contaminate extracted PNGs.
          
          This circumstance should be avoided at any cost.

          Otherwise, it requires human judgment to **de**-contaminate the PNG files.
          
          And the process cannot be **full-auto**.  
          
       2. "False Negative" in the confusion matrix is **acceptable**, cause it will only reduce the number of extracted PNG files.
      
          Once the process is **full-auto**, it is easy to increase the size of extracted PNGs.
          
          There are tons of automatic solutions to this issue.
          
          Ex: Collect more unlabeled images, upgrade the hardware of devices running this code, spend more time to extract PNGs...
	  
  2. Segmentation results from SAM
     
     <img src="/06.Test%20Result/Decoder%20Result.png" height="200">

     - `Green B.Box` : Prediction results from the pre-trained model with overcritical threshold
     - `Green Mask ` : Prediction results from SAM with `Green B.Box` being the prompt
     - `Red   B.Box` : The B.Box for `Green Mask`
     - Comparison :
       
       The `Red B.Box` is more suitable than the `Green B.Box`.
       
       This proves that,
       
       1. any pre-trained model without fine-tuning works properly with an overcritical threshold.
      
       2. SAM did refine the unprecise b.box prompts
     
## 4. Why I store HSV with PNG files
	...
  
## 5. What Next
  - ### Auto Labeled Data Synthesis
  - ### Self-Trained AI-CCTV


