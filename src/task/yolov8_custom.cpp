
#include "yolov8_custom.h"

#include <random>

#include "utils/logging.h"
#include "process/preprocess.h"
#include "process/postprocess.h"

static int g_max_num_output = 20;

// 类别
static std::vector<std::string> g_classes = {
    "plate"};

// 偏移解码
void letterbox_decode(std::vector<Detection> &objects, bool hor, int pad)
{
    for (auto &obj : objects)
    {
        if (hor)
        {
            obj.box.x -= pad;
        }
        else
        {
            obj.box.y -= pad;
        }
    }
}
// 偏移解码
void letterbox_pose_decode(std::vector<std::map<int, KeyPoint>> &keypoints, bool hor, int pad)
{
    for (auto &keypoint : keypoints)
    {
        for (auto &keypoint_item : keypoint)
        {
            if (hor)
            {
                keypoint_item.second.x -= pad;
            }
            else
            {
                keypoint_item.second.y -= pad;
            }
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

    // 预处理包含：letterbox、归一化、BGR2RGB、NCWH
    // 其中RKNN会做：归一化、NCWH转换（详见课程文档），所以这里只需要做letterbox、BGR2RGB
    // 比例
    float wh_ratio = (float)input_tensor_.attr.dims[2] / (float)input_tensor_.attr.dims[1];

    // lettorbox
    if (process_type == "opencv")
    {
        // BGR2RGB，resize，再放入input_tensor_中
        letterbox_info_ = letterbox(img, image_letterbox, wh_ratio);
        cvimg2tensor(image_letterbox, input_tensor_.attr.dims[2], input_tensor_.attr.dims[1], input_tensor_);
    }
    else if (process_type == "rga")
    {
        // rga resize
        letterbox_info_ = letterbox_rga(img, image_letterbox, wh_ratio);
        cvimg2tensor_rga(image_letterbox, input_tensor_.attr.dims[2], input_tensor_.attr.dims[1], input_tensor_);
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
    // 预处理，支持opencv或rga
    auto ret = Preprocess(img, "opencv", image_letterbox);

    // // save image_letterbox
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
    // 偏移
    letterbox_decode(objects, letterbox_info_.hor, letterbox_info_.pad);
    if (model_type_ == NN_YOLOV8_POSE)
        letterbox_pose_decode(keypoints, letterbox_info_.hor, letterbox_info_.pad);

    return NN_SUCCESS;
}