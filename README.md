# Auto-PNG-Extractor
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

## 2. Quick Start
  1. Download the whole project file
  2. Make sure the Python version `python -V` is greater than 3.10
  3. `pip install -r requirement.txt`
  4. `Run 01.SAM Test.py` in [02.Main Code](/02.Main%20Code) → You should get results in [3.Sub Functions](#3-steps)

## 3. Sub Functions
  1. Build B.Box prompts for SAM (any pre-trained model should work, here I use Yolo V8)
     
  2. Segmentation results from SAM

## 4. Why I store HSV with PNG files
	...
  
## 5. What to update
  - ### Current Method :
      ...  
  - ### Improvement :
      ...


