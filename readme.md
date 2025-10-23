### 项目说明
    这个项目基于RK3588的车牌检测+识别部署代码

    - 车牌检测使用的是 CCPD CRPD数据集训练的 yolov8-pose-n， 使用的是4个3维点 [x,y,visable]
    - 车牌识别使用的是开源的onnx

    两个rknn均进行了int8量化

    执行流程：

```
    git clone https://github.com/gaganotlow/plate_recognize.git

    cd plate_recognize

    mkdir build && cd build && cmake .. && make

    ./yolov8_pose_img ../weights/best_relu.rknn ../weights/PlateRec_CTC.rknn ../media/test.jpg

```
<img width="832" height="454" alt="image" src="https://github.com/user-attachments/assets/2320d620-03ed-4c2c-9304-57d553233a54" />


```
推理时间: 37ms
检测到车牌: 3 个
车牌 #1:
推理时间: 3ms
识别结果: 粤CMM281
车牌 #2:
推理时间: 2ms
识别结果: 川A1P68B
车牌 #3:
推理时间: 2ms
识别结果: 川A36B80
共识别 3 个车牌

========================================
推理完成！结果已保存到 result.jpg
========================================
```

