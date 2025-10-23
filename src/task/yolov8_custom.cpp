
#include "yolov8_custom.h"

#include <random>

#include "utils/logging.h"
#include "process/preprocess.h"
#include "process/postprocess.h"

static int g_max_num_output = 20;

// 类别
static std::vector<std::string> g_classes = {
    "plate"};

// 检测框坐标还原：从letterbox坐标系还原到原图坐标系
void letterbox_decode(std::vector<Detection> &objects, const LetterBoxInfo &info)
{
    for (auto &obj : objects)
    {
        // 使用精确的scale_x和scale_y分别还原x和y坐标（修复对齐导致的偏差）
        obj.box.x = (obj.box.x - info.x_pad) / info.scale_x;
        obj.box.y = (obj.box.y - info.y_pad) / info.scale_y;
        obj.box.width = obj.box.width / info.scale_x;
        obj.box.height = obj.box.height / info.scale_y;
    }
}

// 关键点坐标还原：从letterbox坐标系还原到原图坐标系
void letterbox_pose_decode(std::vector<std::map<int, KeyPoint>> &keypoints, const LetterBoxInfo &info)
{
    for (auto &keypoint : keypoints)
    {
        for (auto &keypoint_item : keypoint)
        {
            // 使用精确的scale_x和scale_y分别还原x和y坐标（修复对齐导致的偏差）
            keypoint_item.second.x = (keypoint_item.second.x - info.x_pad) / info.scale_x;
            keypoint_item.second.y = (keypoint_item.second.y - info.y_pad) / info.scale_y;
        }
    }
}

// 构造函数
Yolov8Custom::Yolov8Custom(nn_model_type_e model_type)
{
    engine_ = CreateRKNNEngine();
    input_tensor_.data = nullptr;
    want_float_ = false;
    ready_ = false;
    model_type_ = model_type;
}
// 析构函数
Yolov8Custom::~Yolov8Custom()
{
    // release input tensor and output tensor
    NN_LOG_DEBUG("release input tensor");
    if (input_tensor_.data != nullptr)
    {
        free(input_tensor_.data);
        input_tensor_.data = nullptr;
    }
    NN_LOG_DEBUG("release output tensor");
    for (auto &tensor : output_tensors_)
    {
        if (tensor.data != nullptr)
        {
            free(tensor.data);
            tensor.data = nullptr;
        }
    }
}
// 加载模型
nn_error_e Yolov8Custom::LoadModel(const char *model_path)
{
    auto ret = engine_->LoadModelFile(model_path);
    if (ret != NN_SUCCESS)
    {
        NN_LOG_ERROR("yolov8 load model file failed");
        return ret;
    }
    // get input tensor
    auto input_shapes = engine_->GetInputShapes();

    // check number of input and n_dims
    if (input_shapes.size() != 1)
    {
        NN_LOG_ERROR("yolov8 input tensor number is not 1, but %ld", input_shapes.size());
        return NN_RKNN_INPUT_ATTR_ERROR;
    }
    nn_tensor_attr_to_cvimg_input_data(input_shapes[0], input_tensor_);
    input_tensor_.data = malloc(input_tensor_.attr.size);

    auto output_shapes = engine_->GetOutputShapes();
    // if (output_shapes.size() != 6)
    // {
    //     NN_LOG_ERROR("yolov8 output tensor number is not 9, but %ld", output_shapes.size());
    //     return NN_RKNN_OUTPUT_ATTR_ERROR;
    // }
    if (output_shapes[0].type == NN_TENSOR_FLOAT16)
    {
        want_float_ = true;
        NN_LOG_WARNING("yolov8 output tensor type is float16, want type set to float32");
    }
    for (int i = 0; i < output_shapes.size(); i++)
    {
        tensor_data_s tensor;
        tensor.attr.n_elems = output_shapes[i].n_elems;
        tensor.attr.n_dims = output_shapes[i].n_dims;
        for (int j = 0; j < output_shapes[i].n_dims; j++)
        {
            tensor.attr.dims[j] = output_shapes[i].dims[j];
        }
        // output tensor needs to be float32
        tensor.attr.type = want_float_ ? NN_TENSOR_FLOAT : output_shapes[i].type;
        tensor.attr.index = 0;
        tensor.attr.size = output_shapes[i].n_elems * nn_tensor_type_to_size(tensor.attr.type);
        tensor.data = malloc(tensor.attr.size);
        output_tensors_.push_back(tensor);
        out_zps_.push_back(output_shapes[i].zp);
        out_scales_.push_back(output_shapes[i].scale);
    }

    ready_ = true;
    return NN_SUCCESS;
}
// 预处理
nn_error_e Yolov8Custom::Preprocess(const cv::Mat &img, const std::string process_type, cv::Mat &image_letterbox)
{
    // 预处理包含：letterbox、归一化、BGR2RGB、NCHW
    // 其中RKNN会做：归一化、NCHW转换，所以这里只需要做letterbox、BGR2RGB
    
    int model_width = input_tensor_.attr.dims[2];
    int model_height = input_tensor_.attr.dims[1];
    float wh_ratio = (float)model_width / (float)model_height;

    if (process_type == "opencv")
    {
        // OpenCV版本：CPU处理，letterbox+resize
        letterbox_info_ = letterbox(img, image_letterbox, wh_ratio);
        cvimg2tensor(image_letterbox, model_width, model_height, input_tensor_);
    }
    else if (process_type == "rga")
    {
        // RGA版本：硬件加速，一步完成 letterbox + resize + BGR2RGB
        letterbox_info_ = letterbox_rga(img, model_width, model_height, input_tensor_, 114);
        // 使用tensor.data创建image_letterbox Mat（用于后续处理，如Postprocess）
        image_letterbox = cv::Mat(model_height, model_width, CV_8UC3, input_tensor_.data);
    }
    else
    {
        NN_LOG_ERROR("Unknown preprocess type: %s, use 'opencv' or 'rga'", process_type.c_str());
        return NN_RKNN_INPUT_ATTR_ERROR;
    }

    return NN_SUCCESS;
}
// 推理
nn_error_e Yolov8Custom::Inference()
{
    std::vector<tensor_data_s> inputs;
    inputs.push_back(input_tensor_);
    return engine_->Run(inputs, output_tensors_, want_float_);
}
// 后处理
nn_error_e Yolov8Custom::Postprocess(const cv::Mat &img, std::vector<Detection> &objects, std::vector<std::map<int, KeyPoint>> &keypoints)
{
    void *output_data[g_max_num_dims];
    for (int i = 0; i < output_tensors_.size(); i++)
    {
        output_data[i] = (void *)output_tensors_[i].data;
    }
    std::vector<float> DetectiontRects;

    if (want_float_)
        // 浮点版本
        yolo::GetConvDetectionResult((float **)output_data, DetectiontRects, model_type_, keypoints);
    else
        // 整型版本
        yolo::GetConvDetectionResultInt8((int8_t **)output_data, out_zps_, out_scales_, DetectiontRects, model_type_, keypoints);

    if (model_type_ == NN_YOLOV8_POSE)
    {
        for (auto &kp : keypoints)
        {
            for (auto &kp_item : kp)
            {
                kp_item.second.x = kp_item.second.x * float(img.cols);
                kp_item.second.y = kp_item.second.y * float(img.rows);
            }
        }
    }

    int img_width = img.cols;
    int img_height = img.rows;
    for (int i = 0; i < DetectiontRects.size(); i += 6)
    {
        int classId = int(DetectiontRects[i + 0]);
        float conf = DetectiontRects[i + 1];
        int xmin = int(DetectiontRects[i + 2] * float(img_width) + 0.5);
        int ymin = int(DetectiontRects[i + 3] * float(img_height) + 0.5);
        int xmax = int(DetectiontRects[i + 4] * float(img_width) + 0.5);
        int ymax = int(DetectiontRects[i + 5] * float(img_height) + 0.5);
        Detection result;
        result.class_id = classId;
        result.confidence = conf;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255);
        result.color = cv::Scalar(dis(gen),
                                  dis(gen),
                                  dis(gen));

        result.className = g_classes[result.class_id];
        result.box = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);

        objects.push_back(result);
    }

    return NN_SUCCESS;
}

// 推理
nn_error_e Yolov8Custom::Run(const cv::Mat &img, std::vector<Detection> &objects, std::vector<std::map<int, KeyPoint>> &keypoints)
{
    if (!ready_)
    {
        NN_LOG_ERROR("yolov8 model is not ready");
        return NN_RKNN_MODEL_NOT_LOAD;
    }

    cv::Mat image_letterbox;
 
    // 计时
    auto start = std::chrono::high_resolution_clock::now();
    // 预处理：opencv(CPU) 或 rga(硬件加速，推荐)
    auto ret = Preprocess(img, "rga", image_letterbox);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    NN_LOG_INFO("yolov8 preprocess time: %ldms", duration);

    // // save image_letterbox for debugging
    // cv::imwrite("letterbox.jpg", image_letterbox);

    if (ret != NN_SUCCESS)
    {
        NN_LOG_ERROR("yolov8 preprocess failed");
        return ret;
    }
    ret = Inference();
    if (ret != NN_SUCCESS)
    {
        NN_LOG_ERROR("yolov8 inference failed");
        return ret;
    }
    // 后处理
    ret = Postprocess(image_letterbox, objects, keypoints);
    if (ret != NN_SUCCESS)
    {
        NN_LOG_ERROR("yolov8 postprocess failed");
        return ret;
    }
    
    // 坐标还原：从letterbox坐标系还原到原图坐标系
    NN_LOG_INFO("Before decode: scale=%.3f, x_pad=%d, y_pad=%d", 
                letterbox_info_.scale, letterbox_info_.x_pad, letterbox_info_.y_pad);
    if (!objects.empty()) {
        NN_LOG_INFO("Before decode: box[0]=(%d,%d,%d,%d)", 
                    objects[0].box.x, objects[0].box.y, objects[0].box.width, objects[0].box.height);
    }
    
    letterbox_decode(objects, letterbox_info_);
    if (model_type_ == NN_YOLOV8_POSE)
        letterbox_pose_decode(keypoints, letterbox_info_);
    
    if (!objects.empty()) {
        NN_LOG_INFO("After decode: box[0]=(%d,%d,%d,%d)", 
                    objects[0].box.x, objects[0].box.y, objects[0].box.width, objects[0].box.height);
    }

    return NN_SUCCESS;
}