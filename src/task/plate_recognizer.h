#ifndef RK3588_DEMO_PLATE_RECOGNIZER_H
#define RK3588_DEMO_PLATE_RECOGNIZER_H

#include "engine/engine.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 车牌识别结果结构
struct PlateResult
{
    std::string plate_name;  // 识别的车牌号
    float confidence;        // 置信度（可选）
};

// 车牌识别类，模仿 Yolov8Custom 架构
class PlateRecognizer
{
public:
    PlateRecognizer();
    ~PlateRecognizer();

    // 加载模型
    nn_error_e LoadModel(const char *model_path);
    
    // 推理接口
    nn_error_e Run(const cv::Mat &img, PlateResult &result);

private:
    // 预处理
    nn_error_e Preprocess(const cv::Mat &img);
    
    // 推理
    nn_error_e Inference();
    
    // 后处理（CTC解码）
    nn_error_e Postprocess(PlateResult &result);

    bool ready_;  // 模型是否已加载
    
    // 张量数据
    tensor_data_s input_tensor_;
    std::vector<tensor_data_s> output_tensors_;
    
    // 量化参数
    bool want_float_;
    std::vector<int32_t> out_zps_;
    std::vector<float> out_scales_;
    
    // RKNN引擎
    std::shared_ptr<NNEngine> engine_;
    
    // 车牌字符集
    static const std::vector<std::string> plate_code_;
};

#endif // RK3588_DEMO_PLATE_RECOGNIZER_H

