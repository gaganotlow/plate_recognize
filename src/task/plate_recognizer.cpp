#include "plate_recognizer.h"
#include "utils/logging.h"
#include <algorithm>

// 车牌字符集定义
const std::vector<std::string> PlateRecognizer::plate_code_ = {
    "-",  // CTC空白类 (索引0)
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z",
    "云", "京", "冀", "吉", "鄂", "川", "赣", "新", "贵", "晋", "沪", "津", "浙", "渝", "湘", "琼", "甘", "皖", "粤",
    "辽", "藏", "蒙", "黑", "苏", "鲁", "闽", "宁", "青", "陕", "桂", "豫",
    "学", "港", "澳", "警", "民", "航", "领", "使", "应", "急", "民航", "电"
};

// 构造函数
PlateRecognizer::PlateRecognizer()
{
    engine_ = CreateRKNNEngine();
    input_tensor_.data = nullptr;
    want_float_ = false;
    ready_ = false;
}

// 析构函数
PlateRecognizer::~PlateRecognizer()
{
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
nn_error_e PlateRecognizer::LoadModel(const char *model_path)
{
    auto ret = engine_->LoadModelFile(model_path);
    if (ret != NN_SUCCESS)
    {
        NN_LOG_ERROR("plate recognizer load model file failed");
        return ret;
    }
    
    // 获取输入张量信息
    auto input_shapes = engine_->GetInputShapes();
    if (input_shapes.size() != 1)
    {
        NN_LOG_ERROR("plate recognizer input tensor number is not 1, but %ld", input_shapes.size());
        return NN_RKNN_INPUT_ATTR_ERROR;
    }
    
    // 设置输入张量（期望是 1x3x50x200 或类似格式）
    nn_tensor_attr_to_cvimg_input_data(input_shapes[0], input_tensor_);
    input_tensor_.data = malloc(input_tensor_.attr.size);
    
    NN_LOG_DEBUG("Plate input shape: %d x %d x %d x %d", 
                 input_tensor_.attr.dims[0], input_tensor_.attr.dims[1],
                 input_tensor_.attr.dims[2], input_tensor_.attr.dims[3]);
    
    // 获取输出张量信息
    auto output_shapes = engine_->GetOutputShapes();
    if (output_shapes.size() != 1)
    {
        NN_LOG_WARNING("plate recognizer output tensor number is %ld, expected 1", output_shapes.size());
    }
    
    // 检查是否需要浮点输出
    if (output_shapes[0].type == NN_TENSOR_FLOAT16)
    {
        want_float_ = true;
        NN_LOG_WARNING("plate recognizer output tensor type is float16, want type set to float32");
    }
    
    // 分配输出张量
    for (int i = 0; i < output_shapes.size(); i++)
    {
        tensor_data_s tensor;
        tensor.attr.n_elems = output_shapes[i].n_elems;
        tensor.attr.n_dims = output_shapes[i].n_dims;
        for (int j = 0; j < output_shapes[i].n_dims; j++)
        {
            tensor.attr.dims[j] = output_shapes[i].dims[j];
        }
        
        tensor.attr.type = want_float_ ? NN_TENSOR_FLOAT : output_shapes[i].type;
        tensor.attr.index = 0;
        tensor.attr.size = output_shapes[i].n_elems * nn_tensor_type_to_size(tensor.attr.type);
        tensor.data = malloc(tensor.attr.size);
        
        output_tensors_.push_back(tensor);
        out_zps_.push_back(output_shapes[i].zp);
        out_scales_.push_back(output_shapes[i].scale);
        
        NN_LOG_DEBUG("Plate output[%d] shape: dims=%d, elems=%d", i, tensor.attr.n_dims, tensor.attr.n_elems);
    }
    
    ready_ = true;
    return NN_SUCCESS;
}

// 预处理
nn_error_e PlateRecognizer::Preprocess(const cv::Mat &img)
{
    // 车牌识别预处理：
    // 1. Resize到模型输入尺寸 (通常是 200x50)
    // 2. BGR转RGB
    // 3. 数据格式转换
    
    int model_width = input_tensor_.attr.dims[2];   // 通常是 200
    int model_height = input_tensor_.attr.dims[1];  // 通常是 50
    
    cv::Mat img_resized, img_rgb;
    
    // Resize
    cv::resize(img, img_resized, cv::Size(model_width, model_height));
    
    // BGR转RGB
    cv::cvtColor(img_resized, img_rgb, cv::COLOR_BGR2RGB);
    
    // 复制数据到输入张量
    memcpy(input_tensor_.data, img_rgb.data, input_tensor_.attr.size);
    
    return NN_SUCCESS;
}

// 推理
nn_error_e PlateRecognizer::Inference()
{
    std::vector<tensor_data_s> inputs;
    inputs.push_back(input_tensor_);
    return engine_->Run(inputs, output_tensors_, want_float_);
}

// 后处理 - CTC解码
nn_error_e PlateRecognizer::Postprocess(PlateResult &result)
{
    // CTC 解码
    std::vector<int> outputs_idx;
    
    // 获取输出维度信息
    int seq_length = 20;  // 序列长度，默认值
    int num_classes = 78; // 字符类别数，默认值
    
    // 解析输出张量的形状
    if (output_tensors_[0].attr.n_dims == 3)
    {
        // [1, seq_length, num_classes] -> [seq_length, num_classes]
        seq_length = output_tensors_[0].attr.dims[1];
        num_classes = output_tensors_[0].attr.dims[2];
    }
    else if (output_tensors_[0].attr.n_dims == 2)
    {
        // [seq_length, num_classes]
        seq_length = output_tensors_[0].attr.dims[0];
        num_classes = output_tensors_[0].attr.dims[1];
    }
    
    NN_LOG_DEBUG("Output shape: seq_length=%d, num_classes=%d", seq_length, num_classes);
    
    // 贪婪解码：选择每个位置概率最大的字符
    bool is_int8 = (output_tensors_[0].attr.type == NN_TENSOR_INT8);
    
    for (int i = 0; i < seq_length; i++)
    {
        float max_prob = -1e9;
        int max_idx = 0;
        
        for (int j = 0; j < num_classes; j++)
        {
            float prob;
            
            if (is_int8)
            {
                // 处理 int8 量化输出
                int8_t *ptr = (int8_t *)output_tensors_[0].data;
                int8_t quantized_value = ptr[i * num_classes + j];
                // 反量化: (int8_value - zero_point) * scale
                prob = (quantized_value - out_zps_[0]) * out_scales_[0];
            }
            else
            {
                // 处理浮点输出
                float *ptr = (float *)output_tensors_[0].data;
                prob = ptr[i * num_classes + j];
            }
            
            if (prob > max_prob)
            {
                max_prob = prob;
                max_idx = j;
            }
        }
        outputs_idx.push_back(max_idx);
    }
    
    // CTC解码：去除重复字符和空白类
    std::vector<int> decoded_indices;
    int prev = 0;
    
    for (int i = 0; i < outputs_idx.size(); i++)
    {
        // 去除重复字符和空白类(索引0)
        if (outputs_idx[i] != 0 && outputs_idx[i] != prev)
        {
            decoded_indices.push_back(outputs_idx[i]);
        }
        prev = outputs_idx[i];
    }
    
    // 转换为字符串
    result.plate_name.clear();
    for (int idx : decoded_indices)
    {
        if (idx < plate_code_.size())
        {
            result.plate_name += plate_code_[idx];
        }
    }
    
    result.confidence = 1.0f;  // 可以根据需要计算实际置信度
    
    return NN_SUCCESS;
}

// 运行推理
nn_error_e PlateRecognizer::Run(const cv::Mat &img, PlateResult &result)
{
    if (!ready_)
    {
        NN_LOG_ERROR("plate recognizer model is not ready");
        return NN_RKNN_MODEL_NOT_LOAD;
    }
    
    // 预处理
    auto ret = Preprocess(img);
    if (ret != NN_SUCCESS)
    {
        NN_LOG_ERROR("plate recognizer preprocess failed");
        return ret;
    }
    
    // 推理
    ret = Inference();
    if (ret != NN_SUCCESS)
    {
        NN_LOG_ERROR("plate recognizer inference failed");
        return ret;
    }
    
    // 后处理
    ret = Postprocess(result);
    if (ret != NN_SUCCESS)
    {
        NN_LOG_ERROR("plate recognizer postprocess failed");
        return ret;
    }
    
    return NN_SUCCESS;
}

