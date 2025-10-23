// 预处理

#include "preprocess.h"

#include "utils/logging.h"
#include "im2d.h"
#include "rga.h"

// opencv 版本的 letterbox
LetterBoxInfo letterbox(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio)
{

    // img has to be 3 channels
    if (img.channels() != 3)
    {
        NN_LOG_ERROR("img has to be 3 channels");
        exit(-1);
    }
    float img_width = img.cols;
    float img_height = img.rows;

    int letterbox_width = 0;
    int letterbox_height = 0;

    LetterBoxInfo info;
    memset(&info, 0, sizeof(info));
    
    int padding_hor = 0;
    int padding_ver = 0;

    if (img_width / img_height > wh_ratio)
    {
        info.hor = false;
        letterbox_width = img_width;
        letterbox_height = img_width / wh_ratio;
        info.pad = (letterbox_height - img_height) / 2.f;
        padding_hor = 0;
        padding_ver = info.pad;
        
        // 填充新字段
        info.scale = 1.0f;
        info.scale_x = 1.0f;
        info.scale_y = 1.0f;
        info.x_pad = 0;
        info.y_pad = info.pad;
    }
    else
    {
        info.hor = true;
        letterbox_width = img_height * wh_ratio;
        letterbox_height = img_height;
        info.pad = (letterbox_width - img_width) / 2.f;
        padding_hor = info.pad;
        padding_ver = 0;
        
        // 填充新字段
        info.scale = 1.0f;
        info.scale_x = 1.0f;
        info.scale_y = 1.0f;
        info.x_pad = info.pad;
        info.y_pad = 0;
    }
    
    info.resize_w = letterbox_width;
    info.resize_h = letterbox_height;
    
    /*
     * Padding an image.
                                    dst_img
        --------------      ----------------------------
        |            |      |       top_border         |
        |  src_image |  =>  |                          |
        |            |      |      --------------      |
        --------------      |left_ |            |right_|
                            |border|  dst_rect  |border|
                            |      |            |      |
                            |      --------------      |
                            |       bottom_border      |
                            ----------------------------
     */
    // 使用cv::copyMakeBorder函数进行填充边界
    cv::copyMakeBorder(img, img_letterbox, padding_ver, padding_ver, padding_hor, padding_hor, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return info;
}

// opencv resize
void cvimg2tensor(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor)
{
    // img has to be 3 channels
    if (img.channels() != 3)
    {
        NN_LOG_ERROR("img has to be 3 channels");
        exit(-1);
    }
    // BGR to RGB
    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    // resize img
    cv::Mat img_resized;
    // resize img
    cv::resize(img_rgb, img_resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    // BGR to RGB
    memcpy(tensor.data, img_resized.data, tensor.attr.size);
}

// RGA版本：letterbox + resize + BGR2RGB，直接输出到tensor（一步完成）
LetterBoxInfo letterbox_rga(const cv::Mat &img, int target_width, int target_height, tensor_data_s &tensor, int bg_color)
{
    // img has to be 3 channels
    if (img.channels() != 3)
    {
        NN_LOG_ERROR("img has to be 3 channels");
        exit(-1);
    }
    
    int src_w = img.cols;
    int src_h = img.rows;
    int dst_w = target_width;
    int dst_h = target_height;
    
    int resize_w = dst_w;
    int resize_h = dst_h;
    int padding_w = 0;
    int padding_h = 0;
    int left_offset = 0;
    int top_offset = 0;
    float scale = 1.0;
    
    LetterBoxInfo info;
    memset(&info, 0, sizeof(info));
    
    // 计算缩放比例（选择较小的比例以确保图像完全放入）
    float scale_w = (float)dst_w / src_w;
    float scale_h = (float)dst_h / src_h;
    
    if (scale_w < scale_h)
    {
        scale = scale_w;
        resize_h = (int)(src_h * scale);
    }
    else
    {
        scale = scale_h;
        resize_w = (int)(src_w * scale);
    }
    
    // 对齐处理（RGA硬件要求）
    // RGB888格式：resize_w 必须对齐到16的倍数
    if (resize_w % 16 != 0)
    {
        resize_w -= resize_w % 16;
    }
    // resize_h 对齐到2的倍数
    if (resize_h % 2 != 0)
    {
        resize_h -= resize_h % 2;
    }
    
    // 计算padding
    padding_h = dst_h - resize_h;
    padding_w = dst_w - resize_w;
    
    // 居中padding
    if (scale_w < scale_h)
    {
        info.hor = false;
        top_offset = padding_h / 2;
        // top_offset 对齐到2的倍数
        if (top_offset % 2 != 0)
        {
            top_offset -= top_offset % 2;
            if (top_offset < 0)
            {
                top_offset = 0;
            }
        }
        left_offset = 0;
        info.pad = top_offset;
    }
    else
    {
        info.hor = true;
        left_offset = padding_w / 2;
        // left_offset 对齐到2的倍数
        if (left_offset % 2 != 0)
        {
            left_offset -= left_offset % 2;
            if (left_offset < 0)
            {
                left_offset = 0;
            }
        }
        top_offset = 0;
        info.pad = left_offset;
    }
    
    // 保存letterbox信息
    info.scale = scale;  // 兼容旧代码
    // 使用对齐后的实际尺寸计算精确的scale（修复对齐导致的坐标偏差）
    info.scale_x = (float)resize_w / src_w;
    info.scale_y = (float)resize_h / src_h;
    info.x_pad = left_offset;
    info.y_pad = top_offset;
    info.resize_w = resize_w;
    info.resize_h = resize_h;
    
    NN_LOG_INFO("RGA Letterbox: src(%d,%d) -> dst(%d,%d), resize(%d,%d), scale=%.3f, offset(%d,%d)",
                src_w, src_h, dst_w, dst_h, resize_w, resize_h, scale, left_offset, top_offset);
    
    // 使用tensor的内存创建cv::Mat（避免额外内存分配和拷贝）
    cv::Mat letterbox_mat(dst_h, dst_w, CV_8UC3, tensor.data);
    
    // 填充背景色（RGB格式）
    letterbox_mat.setTo(cv::Scalar(bg_color, bg_color, bg_color));
    
    // 使用RGA进行resize + BGR2RGB格式转换（一步完成）
    cv::Mat img_resized = cv::Mat::zeros(resize_h, resize_w, CV_8UC3);
    
    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));
    
    // RGA: resize + BGR2RGB格式转换
    // 输入：BGR888，输出：RGB888
    rga_buffer_t src = wrapbuffer_virtualaddr((void *)img.data, img.cols, img.rows, RK_FORMAT_BGR_888);
    rga_buffer_t dst_resize = wrapbuffer_virtualaddr((void *)img_resized.data, resize_w, resize_h, RK_FORMAT_RGB_888);
    
    int ret = imcheck(src, dst_resize, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret)
    {
        NN_LOG_ERROR("%d, RGA resize check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        exit(-1);
    }
    
    // imresize会自动处理BGR到RGB的转换
    ret = imresize(src, dst_resize);
    if (ret != IM_STATUS_SUCCESS)
    {
        NN_LOG_ERROR("%d, RGA resize failed! %s", __LINE__, imStrError((IM_STATUS)ret));
        exit(-1);
    }
    
    // 将resize后的RGB图像复制到tensor的letterbox中心位置
    cv::Rect roi(left_offset, top_offset, resize_w, resize_h);
    img_resized.copyTo(letterbox_mat(roi));
    
    return info;
}