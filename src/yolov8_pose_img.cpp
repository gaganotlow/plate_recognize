
#include <opencv2/opencv.hpp>
#include "task/yolov8_custom.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"

int main(int argc, char **argv)
{
    // 模型类型
    nn_model_type_e model_type = NN_YOLOV8_POSE;
    Yolov8Custom yolo(model_type);
    // 模型地址
    const char *model_file = argv[1];
    // 输入图片地址
    const char *img_file = argv[2];
    // 加载图片
    cv::Mat img = cv::imread(img_file, cv::IMREAD_COLOR);
    // 加载模型
    yolo.LoadModel(model_file);
    // 检测框
    std::vector<Detection> objects;
    // 关键点
    std::vector<std::map<int, KeyPoint>> kps;
    // 推理
        auto start = std::chrono::high_resolution_clock::now();
        yolo.Run(img, objects, kps);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "推理时间: " << duration.count() << "ms" << std::endl;

        // 绘制
        DrawDetections(img, objects);
        if (model_type == NN_YOLOV8_POSE)
        {
            DrawCocoKps(img, kps);
        }
        cv::imwrite("result.jpg", img);

    return 0;
}