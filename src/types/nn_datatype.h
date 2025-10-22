
#ifndef RK3588_DEMO_NN_DATATYPE_H
#define RK3588_DEMO_NN_DATATYPE_H

#include <opencv2/opencv.hpp>

typedef struct _nn_object_s {
    float x;
    float y;
    float w;
    float h;
    float score;
    int class_id;
} nn_object_s;

struct Detection
{
    int class_id{0};
    std::string className{};
    float confidence{0.0};
    cv::Scalar color{};
    cv::Rect box{};
};

typedef enum {
    NN_YOLOV8_DET = 0,
    NN_YOLOV8_SEG = 1,
    NN_YOLOV8_POSE = 2
}nn_model_type_e;

typedef struct
{
    float x;
    float y;
    float score;
    int id;
} KeyPoint;
typedef struct
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    int classId;
    float mask[32];
    std::vector<KeyPoint> keyPoints;
} DetectRect;

#endif //RK3588_DEMO_NN_DATATYPE_H
