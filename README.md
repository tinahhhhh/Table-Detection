# Swin Transformer for Table Detection

This repo is based on [Swin Transformer for Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection). A Faster RCNN with [Swin Transformer](https://arxiv.org/pdf/2103.14030.pdf) backbone config file, dataset and a script for transforming dataset into COCO format are included.

## How to use
Run [Swin\_Transformer\_Table\_Detection.ipynb](https://github.com/tinahhhhh/Table-Detection/blob/master/Swin_Transformer_Table_Detection.ipynb). 

 <a href="https://colab.research.google.com/github/tinahhhhh/Table-Detection/blob/master/Swin_Transformer_Table_Detection.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Results

10 epochs:  
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.854  
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.938  
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.929  
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000  
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000  
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.854  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.896  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.896  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.896  
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000  
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.896   

Result from cTDaR_t10499.jpg  

<img src="imgs/detection.png"/>  

Note:  
Fine-tune cfg.runner.max\_epochs and cfg.checkpoint\_config.interval to change the number of epochs. (In Swin\_Transformer\_Table\_Detection.ipynb, I set them to 3.)

## Dataset
[cTDaR TRACKA](https://zenodo.org/record/2649217#.YInpcX0zZTZ)

## References
1. [Voc2coco](https://github.com/yukkyo/voc2coco)

2. [MMDetection document](https://mmdetection.readthedocs.io/en/latest/)