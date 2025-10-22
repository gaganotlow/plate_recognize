
#ifndef RK3588_DEMO_POSTPROCESS_H
#define RK3588_DEMO_POSTPROCESS_H

#include <stdint.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "types/nn_datatype.h"

int get_top(float *pfProb, float *pfMaxProb, uint32_t *pMaxClass, uint32_t outputCount, uint32_t topNum);

namespace yolo
{
    // int8版本
    int GetConvDetectionResultInt8(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale,
                                   std::vector<float> &DetectiontRects, nn_model_type_e model_type,
                                   std::vector<std::map<int, KeyPoint>> &keypoints);
    // 浮点版本
    int GetConvDetectionResult(float **pBlob, std::vector<float> &DetectiontRects, nn_model_type_e model_type,
                               std::vector<std::map<int, KeyPoint>> &keypoints);

}

#endif // RK3588_DEMO_POSTPROCESS_H
