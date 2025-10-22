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
