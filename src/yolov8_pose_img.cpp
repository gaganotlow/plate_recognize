
#include <opencv2/opencv.hpp>
#include "task/yolov8_custom.h"
#include "task/plate_recognizer.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"

// 关键点排序：按照（左上，右上，右下，左下）的顺序排列
std::vector<cv::Point2f> order_points(const std::vector<cv::Point2f> &pts)
{
    std::vector<cv::Point2f> rect(4);
    
    // 计算和
    std::vector<float> sums(pts.size());
    for (int i = 0; i < pts.size(); i++) {
        sums[i] = pts[i].x + pts[i].y;
    }
    
    // 和最小的是左上角，和最大的是右下角
    int min_idx = std::min_element(sums.begin(), sums.end()) - sums.begin();
    int max_idx = std::max_element(sums.begin(), sums.end()) - sums.begin();
    rect[0] = pts[min_idx];  // 左上
    rect[2] = pts[max_idx];  // 右下
    
    // 计算差值
    std::vector<float> diffs(pts.size());
    for (int i = 0; i < pts.size(); i++) {
        diffs[i] = pts[i].y - pts[i].x;
    }
    
    // 差值最小的是右上角，差值最大的是左下角
    min_idx = std::min_element(diffs.begin(), diffs.end()) - diffs.begin();
    max_idx = std::max_element(diffs.begin(), diffs.end()) - diffs.begin();
    rect[1] = pts[min_idx];  // 右上
    rect[3] = pts[max_idx];  // 左下
    
    return rect;
}

// 透视变换得到矫正后的图像
cv::Mat four_point_transform(const cv::Mat &image, const std::vector<cv::Point2f> &pts)
{
    std::vector<cv::Point2f> rect = order_points(pts);
    cv::Point2f tl = rect[0], tr = rect[1], br = rect[2], bl = rect[3];
    
    // 计算宽度
    float widthA = std::sqrt(std::pow(br.x - bl.x, 2) + std::pow(br.y - bl.y, 2));
    float widthB = std::sqrt(std::pow(tr.x - tl.x, 2) + std::pow(tr.y - tl.y, 2));
    int maxWidth = std::max(int(widthA), int(widthB));
    
    // 计算高度
    float heightA = std::sqrt(std::pow(tr.x - br.x, 2) + std::pow(tr.y - br.y, 2));
    float heightB = std::sqrt(std::pow(tl.x - bl.x, 2) + std::pow(tl.y - bl.y, 2));
    int maxHeight = std::max(int(heightA), int(heightB));
    
    // 目标点
    std::vector<cv::Point2f> dst = {
        cv::Point2f(0, 0),
        cv::Point2f(maxWidth - 1, 0),
        cv::Point2f(maxWidth - 1, maxHeight - 1),
        cv::Point2f(0, maxHeight - 1)
    };
    
    // 透视变换
    cv::Mat M = cv::getPerspectiveTransform(rect, dst);
    cv::Mat warped;
    cv::warpPerspective(image, warped, M, cv::Size(maxWidth, maxHeight));
    
    return warped;
}

// 从关键点提取车牌区域（带扩展）
// expand_ratio_h: 左右扩展比例
// expand_ratio_v: 上下扩展比例
cv::Mat extract_plate_from_keypoints(const cv::Mat &image, const std::map<int, KeyPoint> &keypoints, 
                                     float expand_ratio_h = 0.1, float expand_ratio_v = 0.1)
{
    // 收集关键点坐标
    std::vector<cv::Point2f> pts;
    for (const auto &kp : keypoints) {
        pts.push_back(cv::Point2f(kp.second.x, kp.second.y));
    }
    
    if (pts.size() != 4) {
        NN_LOG_WARNING("Expected 4 keypoints for plate, got %zu", pts.size());
        // 如果不是4个点，返回空图像
        return cv::Mat();
    }
    
    // 排序关键点
    std::vector<cv::Point2f> ordered_pts = order_points(pts);
    
    // 计算扩展
    cv::Point2f tl = ordered_pts[0], tr = ordered_pts[1], br = ordered_pts[2], bl = ordered_pts[3];
    
    float width = std::max(
        std::sqrt(std::pow(tr.x - tl.x, 2) + std::pow(tr.y - tl.y, 2)),
        std::sqrt(std::pow(br.x - bl.x, 2) + std::pow(br.y - bl.y, 2))
    );
    float height = std::max(
        std::sqrt(std::pow(bl.x - tl.x, 2) + std::pow(bl.y - tl.y, 2)),
        std::sqrt(std::pow(br.x - tr.x, 2) + std::pow(br.y - tr.y, 2))
    );
    
    // 分别计算左右和上下的扩展量
    float expand_w = width * expand_ratio_h;   // 左右扩展
    float expand_h = height * expand_ratio_v;  // 上下扩展
    
    // 左右扩展，上下也适当扩展
    std::vector<cv::Point2f> expanded_pts(4);
    
    // 左上角：向左上扩展
    expanded_pts[0] = cv::Point2f(tl.x - expand_w, tl.y - expand_h);
    // 右上角：向右上扩展
    expanded_pts[1] = cv::Point2f(tr.x + expand_w, tr.y - expand_h);
    // 右下角：向右下扩展
    expanded_pts[2] = cv::Point2f(br.x + expand_w, br.y + expand_h);
    // 左下角：向左下扩展
    expanded_pts[3] = cv::Point2f(bl.x - expand_w, bl.y + expand_h);
    
    // 限制在图像范围内
    for (auto &pt : expanded_pts) {
        pt.x = std::max(0.0f, std::min(pt.x, float(image.cols - 1)));
        pt.y = std::max(0.0f, std::min(pt.y, float(image.rows - 1)));
    }
    
    // 透视变换
    return four_point_transform(image, expanded_pts);
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        printf("Usage: %s <pose_model_path> <plate_rec_model_path> <image_path>\n", argv[0]);
        return -1;
    }

    // 模型地址
    const char *pose_model_file = argv[1];
    const char *plate_rec_model_file = argv[2];
    const char *img_file = argv[3];

    // ========== 初始化姿态检测模型 ==========
    nn_model_type_e model_type = NN_YOLOV8_POSE;
    Yolov8Custom yolo(model_type);
    
    auto ret_pose = yolo.LoadModel(pose_model_file);
    if (ret_pose != NN_SUCCESS) {
        NN_LOG_ERROR("Failed to load pose model: %s", pose_model_file);
        return -1;
    }

    // ========== 初始化车牌识别模型 ==========
    PlateRecognizer plate_rec;
    
    auto ret_plate = plate_rec.LoadModel(plate_rec_model_file);
    if (ret_plate != NN_SUCCESS) {
        NN_LOG_ERROR("Failed to load plate recognition model: %s", plate_rec_model_file);
        return -1;
    }

    // ========== 加载图片 ==========
    cv::Mat img = cv::imread(img_file, cv::IMREAD_COLOR);
    if (img.empty()) {
        NN_LOG_ERROR("Failed to load image: %s", img_file);
        return -1;
    }

    // ========== 姿态检测推理 ==========
    std::vector<Detection> objects;
    std::vector<std::map<int, KeyPoint>> kps;
    
    NN_LOG_INFO("Input image size: %dx%d", img.cols, img.rows);
    auto start_pose = std::chrono::high_resolution_clock::now();
    ret_pose = yolo.Run(img, objects, kps);
    auto end_pose = std::chrono::high_resolution_clock::now();
    NN_LOG_INFO("After Run, img size: %dx%d", img.cols, img.rows);
    
    if (ret_pose != NN_SUCCESS) {
        NN_LOG_ERROR("Pose detection failed");
    } else {
        auto duration_pose = std::chrono::duration_cast<std::chrono::milliseconds>(end_pose - start_pose);
        std::cout << "推理时间: " << duration_pose.count() << "ms" << std::endl;
        std::cout << "检测到车牌: " << objects.size() << " 个" << std::endl;
    }

    std::vector<std::string> plate_numbers;  // 存储识别结果，稍后绘制
    
    if (kps.empty()) {
        std::cout << "未检测到车牌关键点，跳过车牌识别" << std::endl;
    } else {
        // 对每个检测到的车牌进行识别
        int plate_count = 0;
        for (size_t i = 0; i < kps.size(); i++) {
            const auto &keypoint_map = kps[i];
            
            // 从原始图像（无绘制痕迹）提取车牌区域
            // 左右扩展10%，上下扩展3%
            cv::Mat plate_roi = extract_plate_from_keypoints(img, keypoint_map, 0.1, 0.2);
            
            if (plate_roi.empty()) {
                NN_LOG_WARNING("Failed to extract plate ROI for object %zu", i);
                plate_numbers.push_back("");  // 占位
                continue;
            }
            
            // 保存裁剪后的车牌图像用于调试
            cv::imwrite("plate_roi_" + std::to_string(i) + ".jpg", plate_roi);
            
            // 车牌识别
            PlateResult plate_result;
            auto start_plate = std::chrono::high_resolution_clock::now();
            ret_plate = plate_rec.Run(plate_roi, plate_result);
            auto end_plate = std::chrono::high_resolution_clock::now();
            
            if (ret_plate != NN_SUCCESS) {
                NN_LOG_ERROR("Plate recognition failed for object %zu", i);
                plate_numbers.push_back("");  // 占位
            } else {
                auto duration_plate = std::chrono::duration_cast<std::chrono::milliseconds>(end_plate - start_plate);
                plate_count++;
                std::cout << "车牌 #" << (i + 1) << ":" << std::endl;
                std::cout << "推理时间: " << duration_plate.count() << "ms" << std::endl;
                std::cout << "识别结果: " << plate_result.plate_name << std::endl;
                plate_numbers.push_back(plate_result.plate_name);
            }
        }
        
        if (plate_count == 0) {
            std::cout << "未成功识别任何车牌" << std::endl;
        } else {
            std::cout << "共识别 " << plate_count << " 个车牌" << std::endl;
        }
    }
    
    // ========== 绘制检测结果 ==========
    if (ret_pose == NN_SUCCESS) {
        DrawDetections(img, objects);
        if (model_type == NN_YOLOV8_POSE) {
            DrawCocoKps(img, kps);
        }
        
        // 在图像上绘制识别的车牌号码
        for (size_t i = 0; i < plate_numbers.size(); i++) {
            if (!plate_numbers[i].empty() && i < objects.size()) {
                cv::Point text_pos(objects[i].box.x, objects[i].box.y - 10);
                cv::putText(img, plate_numbers[i], text_pos,
                           cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            }
        }
    }

    // ========== 保存结果 ==========
    cv::imwrite("result.jpg", img);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "推理完成！结果已保存到 result.jpg" << std::endl;
    std::cout << "========================================\n" << std::endl;

    return 0;
}