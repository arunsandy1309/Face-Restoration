# SSD: Single Shot MultiBox Object Detector - PyTorch Implementation

<img src="https://user-images.githubusercontent.com/50144683/234833819-a7d077ca-c5e5-45a7-8f21-bd3098f61d2c.jpg"></br>
Example SSD output (vgg_ssd300_voc0712).

This repository contains the Pytorch implementation of the following paper:
>**SSD: Single Shot MultiBox Detector**</br>
>Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg</br>
>https://arxiv.org/abs/1512.02325
>
>**Abstract:** _We present a method for detecting objects in images using a single deep neural network. Our approach, named SSD, discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location. At prediction time, the network generates scores for the presence of each object category in each default box and produces adjustments to the box to better match the object shape. Additionally, the network combines predictions from multiple feature maps with different resolutions to naturally handle objects of various sizes. SSD is simple relative to methods that require object proposals because it completely eliminates proposal generation and subsequent pixel or feature resampling stages and encapsulates all computation in a single
network. This makes SSD easy to train and straightforward to integrate into systems that require a detection component. Experimental results on the PASCAL VOC, COCO, and ILSVRC datasets confirm that SSD has competitive accuracy to methods that utilize an additional object proposal step and is much faster, while providing a unified framework for both training and inference. For 300 × 300 input, SSD achieves 74.3% mAP1 on VOC2007 test at 59 FPS on a Nvidia Titan X and for 512 × 512 input, SSD achieves 76.9% mAP, outperforming a comparable state-of-the-art Faster R-CNN model. Compared to other single stage methods, SSD has much better accuracy even with a smaller input image size._

## SSD Framework
<img src="https://user-images.githubusercontent.com/50144683/234837128-70bf7b8f-5bf4-4492-9fe4-a029f5dfeaf1.png"></br>
(a) SSD only needs an input image and ground truth boxes for each object during training. In a convolutional fashion, we evaluate a small set of default boxes of different aspect ratios at each location in several feature maps with different scales (e.g. 8 × 8 and 4 × 4 in (b) and (c)). 
For each default box, we predict both the shape offsets and the confidences for all object categories ((c1, c2, · · · , cp)). At training time, we first match these default boxes to the ground truth boxes. For example, we have matched two default boxes with the cat and one with the dog, which are treated as positives and the rest as negatives. The model loss is a weighted sum between localization loss (e.g. Smooth L1) and confidence loss (e.g. Softmax).

## Architecture
<img src="https://user-images.githubusercontent.com/50144683/234840433-d68d4285-5009-4ff1-82c1-ce9c310737c0.png"></br>
The SSD approach is based on a feed-forward convolutional network that produces a fixed-size collection of bounding boxes and scores for the presence of object class instances in those boxes, followed by a non-maximum suppression step to produce the final detections. The early network layers are based on a standard architecture used for high quality image classification (truncated before any classification layers), which we will call the base network. Our SSD model adds several feature layers to the end of a base network, which predict the offsets to default boxes of different scales and aspect ratios and their associated confidences. SSD with a 300 × 300 input size significantly outperforms its 448 × 448 YOLO counterpart in accuracy on VOC2007 test while also improving the speed.

## Use a pre-trained SSD network for detection
Download a pre-trained network
+ We are trying to provide PyTorch state_dicts (dict of weight tensors) of the latest SSD model definitions trained on different datasets.
+ Currently, we provide the following PyTorch models:
  - SSD300 trained on VOC0712 (newest PyTorch weights)
    - https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth
  - SSD300 trained on VOC0712 (original Caffe weights)
    - https://s3.amazonaws.com/amdegroot-models/ssd_300_VOC0712.pth </br>
    
![image](https://user-images.githubusercontent.com/50144683/234842695-c4797086-6664-4c1e-9e29-cccf254bcfe9.png)

## Usage
The blow script will take the "Funny_dog.mp4" as an input.</br>
```
python Object_Detection.py
```

| Input | Output |
| --------------------------------- | --------------------------------- |
| <video src="https://user-images.githubusercontent.com/50144683/234843854-5fc189c8-34f6-46bc-84ff-8549f8f76cbb.mp4"> | <video src="https://user-images.githubusercontent.com/50144683/234844307-2d42b17d-5b0f-4e49-9179-428ea43fdd07.mp4"> |
