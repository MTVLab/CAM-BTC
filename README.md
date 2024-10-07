# Overview

![CAM](D:\deep_learning\project\zw\GitHub\ConvNext_CAM\CAM.png)

# Environment
python=3.8 pytorch=2.0.1 torchvision=0.15.2 tensorboard=2.14.0 scikit-learn=0.22 pandas=1.3.2

# Reference
ABN [CVPR 2019] https://arxiv.org/pdf/1812.10025.pdf  
LFICAM [ICCV 2021] https://arxiv.org/pdf/2105.00937.pdf  

# Data format
/data/PIC  
|—— BM  
|   ----- |—case1   
|   ----- |—case2  
|—— GBM  
|   ----- |—case1  
|   ----- |—case2  
...  


# Run
Ten fold cross-validate the json file in the /data directory
Click train.py to run the code with more comments
compare contains seven contrast models whose interfaces can be called directly from train.py
predict.py contains inference and plotting code

