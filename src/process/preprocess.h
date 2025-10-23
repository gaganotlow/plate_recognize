// 预处理

#ifndef RK3588_DEMO_PREPROCESS_H
#define RK3588_DEMO_PREPROCESS_H

#include <opencv2/opencv.hpp>
#include "types/datatype.h"

struct LetterBoxInfo
{
    bool hor;           // 是否水平padding
    int pad;            // padding大小（兼容旧代码）
    float scale;        // 缩放比例（兼容旧代码，使用较小的scale）
    float scale_x;      // x方向缩放比例（更精确）
    float scale_y;      // y方向缩放比例（更精确）
    int x_pad;          // 水平padding（左侧）
    int y_pad;          // 垂直padding（上侧）
    int resize_w;       // resize后的宽度（对齐后）
    int resize_h;       // resize后的高度（对齐后）
};

// OpenCV版本：letterbox + BGR2RGB + resize
LetterBoxInfo letterbox(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio);
void cvimg2tensor(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor);

// RGA版本：letterbox + resize + BGR2RGB，直接输出到tensor（一步完成）
LetterBoxInfo letterbox_rga(const cv::Mat &img, int target_width, int target_height, tensor_data_s &tensor, int bg_color = 114);

#endif // RK3588_DEMO_PREPROCESS_H
