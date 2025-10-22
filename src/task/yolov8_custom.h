

#ifndef RK3588_DEMO_YOLOV8_CUSTOM_H
#define RK3588_DEMO_YOLOV8_CUSTOM_H

#include "engine/engine.h"

#include <memory>

#include <opencv2/opencv.hpp>

#include "types/nn_datatype.h"
#include "process/preprocess.h"

class Yolov8Custom
{
public:
    Yolov8Custom(nn_model_type_e model_type = NN_YOLOV8_DET);
    ~Yolov8Custom();

    nn_error_e LoadModel(const char *model_path);
    nn_error_e Run(const cv::Mat &img, std::vector<Detection> &objects, std::vector<std::map<int, KeyPoint>> &keypoints);

private:
    nn_error_e Preprocess(const cv::Mat &img, const std::string process_type, cv::Mat &image_letterbox);
    nn_error_e Inference();
    nn_error_e Postprocess(const cv::Mat &img, std::vector<Detection> &objects, std::vector<std::map<int, KeyPoint>> &keypoints);

    bool ready_;

    nn_model_type_e model_type_;
    LetterBoxInfo letterbox_info_;
    tensor_data_s input_tensor_;
    std::vector<tensor_data_s> output_tensors_;
    bool want_float_;
    std::vector<int32_t> out_zps_;
    std::vector<float> out_scales_;
    std::shared_ptr<NNEngine> engine_;
};

#endif // RK3588_DEMO_YOLOV8_CUSTOM_H
