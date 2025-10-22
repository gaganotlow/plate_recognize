
#include "cv_draw.h"

#include "utils/logging.h"

void DrawDetections(cv::Mat& img, const std::vector<Detection>& objects) {
    NN_LOG_DEBUG("draw %ld objects", objects.size());
    for (const auto& object : objects) {
        cv::rectangle(img, object.box, object.color, 2);
        // class name with confidence
        std::string draw_string = object.className + " " + std::to_string(object.confidence);
        cv::putText(img, draw_string, cv::Point(object.box.x, object.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    object.color, 2);
    }
}

void DrawMask(cv::Mat& img, cv::Mat& seg_mask) {
    cv::addWeighted(img, 0.55, seg_mask, 0.45, 0, img);
}

void DrawCocoKps(cv::Mat& img, const std::vector<std::map<int, KeyPoint>>& keypoints) {
    for (const auto& keypoint : keypoints) {
        for (const auto& keypoint_item : keypoint) {
            cv::circle(img, cv::Point(keypoint_item.second.x, keypoint_item.second.y), 5, cv::Scalar(0, 255, 0), -1);
        }
    }
    // draw skeleton
    // skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    //            [6, 7], [6, 8], [7, 9], [8, 10], [9, 11],
    //            [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    // static const std::vector<std::vector<int>> joint_pairs =
    //             {{16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13},
    //             {6, 7},  {6, 8},  {7, 9},  {8, 10}, {9, 11},
    //             {2, 3},  {1, 2}, {1, 3},  {2, 4},  {3, 5},  {4, 6},  {5, 7}};
    // 正确的关节对定义
    static const std::vector<std::pair<int, int>> joint_pairs = {
        {0, 1},  // 关节0到关节1
        {1, 2},  // 关节1到关节2
        {2, 3},  // 关节2到关节3
        {3, 0}   // 关节3到关节0
    };

    // 绘制关节连接线
    for (const auto& keypoint : keypoints) {
        for (const auto& joint_pair : joint_pairs) {
            int joint_idx1 = joint_pair.first;
            int joint_idx2 = joint_pair.second;
            
            const auto& joint1 = keypoint.find(joint_idx1);
            const auto& joint2 = keypoint.find(joint_idx2);
            
            if (joint1 != keypoint.end() && joint2 != keypoint.end()) {
                cv::Point pt1(joint1->second.x, joint1->second.y);
                cv::Point pt2(joint2->second.x, joint2->second.y);
                
                cv::line(img, pt1, pt2, cv::Scalar(0, 255, 255), 2);
            }
        }
    }
}