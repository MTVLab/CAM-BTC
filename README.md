# 架构

![image](https://github.com/MTVLab/ConvNext_CAM/blob/main/CAM.png)

# 环境
python=3.8 pytorch=2.0.1 torchvision=0.15.2 tensorboard=2.14.0 scikit-learn=0.22 pandas=1.3.2

# 参考文献
ABN [CVPR 2019] https://arxiv.org/pdf/1812.10025.pdf  
LFICAM [ICCV 2021] https://arxiv.org/pdf/2105.00937.pdf  

# 数据格式
/data/PIC  
|—— BM  
|   ----- |—case1   
|   ----- |—case2  
|—— GBM  
|   ----- |—case1  
|   ----- |—case2  
...  


# 运行
十折交叉验证的json文件在/data目录下   
点击train.py即可运行 代码包含较多注释  
compare下包含七个对比模型 可在train.py中直接调用其接口  
predict.py中包含推理和绘图代码  
代码有待优化 了解思路后可自己更改  

