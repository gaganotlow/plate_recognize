

#ifndef RK3588_DEMO_CV_DRAW_H
#define RK3588_DEMO_CV_DRAW_H

#include <opencv2/opencv.hpp>

#include "types/nn_datatype.h"

// draw detections on img
void DrawDetections(cv::Mat& img, const std::vector<Detection>& objects);
void DrawMask(cv::Mat& img, cv::Mat& seg_mask);
void DrawCocoKps(cv::Mat& img, const std::vector<std::map<int, KeyPoint>>& keypoints);

#endif //RK3588_DEMO_CV_DRAW_H
