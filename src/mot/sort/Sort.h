// SORT 跟踪的头文件

#ifndef __SORT_H__
#define __SORT_H__
#include <KalmanFilter.h>

#include <Eigen/Dense>
#include <vector>
using namespace std;

/// @brief SORT 参数
typedef struct _SortParam {
    double iou_threshold = 0.3;  // iou匹配上的阈值
    int max_missed = 10;         // 最大连续未匹配成功的帧数
    int min_hits = 3;            // 最小连续匹配成功的帧数
} SortParam;

/// @brief BoundingBox
using BBox = Eigen::Vector4d;  // [x1,y1,x2,y2] 左上 右下

/// @brief 检测结果
typedef struct _Detection {
    BBox bbox;
    double score;  // 检测框的分数
    int cls_id;    // 检测框对应的目标id
} Detection;

/// @brief 一个独立的被跟踪目标的轨迹类
class Track {
public:
    int id;                // 跟踪轨迹id 转正前<0
    KalmanFilter kf;       // 卡尔曼滤波器
    SortParam sort_param;  // 跟踪参数
    int age = 0;           // 轨迹存在的总帧数
    int hits = 0;          // 连续匹配成功的帧数
    int missed = 0;        // 连续未匹配成功的帧数

    Track(int track_id, const BBox& bbox, SortParam& param) : id(track_id), kf(bbox), sort_param(param) {}
    void update(const BBox& z) {
        kf.update(z);  // KF 更新
        hits++;
        missed = 0;  // 重置未匹配计数
    }
    void predict() {
        kf.predict();  // KF 预测
        age++;
        missed++;
    }
    BBox get_state_bbox() {
        auto bbox = kf.KalmanState2Bbox();
        return {bbox(0), bbox(1), bbox(2), bbox(3)};
    }
    bool is_activate() const {
        // 必须满足最小命中次数，且未超过最大丢失帧数
        return hits >= sort_param.min_hits && missed < sort_param.max_missed;
    }
};

/// @brief SORT 跟踪器管理类 SORT 核心逻辑 (预测、关联、更新、创建、删除)。
class SortTracker {
private:
    KalmanFilterParam m_kf_params;       // 卡尔曼滤波器参数
    SortParam m_sort_params;             // SORT跟踪器参数
    vector<unique_ptr<Track>> m_tracks;  // 存储所有轨迹（使用智能指针管理内存）
    int next_id = -1;                    // 下一个可用轨迹 ID
    int next_activate_id = 1;            // 下一个跟踪转正之后的 ID

    double calculate_iou(const BBox& rect1, const BBox& rect2);
    vector<vector<double>> solve_assignment(const vector<BBox>& pred_boxes, const vector<BBox>& detections,
                                            float iou_threshold);
    vector<vector<double>> predict_and_associate(const vector<BBox> detections);

public:
    SortTracker(const SortParam& sort_params, const KalmanFilterParam& kf_params)
        : m_sort_params(sort_params), m_kf_params(kf_params) {
        // 初始化卡尔曼滤波器的固定参数
        // KalmanFilter::initGlobalParams(kf_params);
        KalmanFilter kf(kf_params);
    };
    vector<pair<int, BBox>> update(const vector<BBox>& detections);
    ~SortTracker() = default;
};

#endif  // __SORT_H__