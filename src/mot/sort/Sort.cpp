#include "Sort.h"
#include "lapjv.h"
#include <spdlog/spdlog.h>


/// @brief  更新函数，处理当前帧的检测结果
/// @param detections 当前帧的边界框检测框列表
/// @return 包含 ID 和边界框的活动轨迹列表
vector<pair<int, BBox>> SortTracker::update(const vector<BBox>& detections) {
    spdlog::info("[SORT] update begin!!!");
    // 1、对已有的跟踪器进行预测和关联
    vector<vector<double>> matched_indices = predict_and_associate(detections);
    spdlog::info("[SORT] predict_and_associate finish!!!");

    // 2. 标记已匹配和未匹配的检测和轨迹
    vector<bool> matched_tracks(m_tracks.size(), false);
    vector<bool> matched_dets(detections.size(), false);
    for (int i = 0; i < matched_indices.size(); ++i) {
        for (int j = 0; j < matched_indices[0].size(); ++j) {
            if (matched_indices[i][j] == 1) {
                m_tracks[i]->update(detections[j]);
                if (m_tracks[i]->is_activate() && m_tracks[i]->id <0) { // 判断激活并且id未转正
                    m_tracks[i]->id = next_activate_id++;  // 符合转正条件的赋予转正id
                }
                matched_tracks[i] = true;
                matched_dets[j] = true;
            }
        }
    }

    // 3. 处理未匹配的检测 (Unmatched Detections) -> 创建新轨迹
    for (int j = 0; j < detections.size(); ++j) {
        if (!matched_dets[j]) {
            // 创建新轨迹，分配新 ID
            m_tracks.emplace_back(make_unique<Track>(next_id--, detections[j], m_sort_params)); // 刚开始分配的时id，符合转正条件的时候转正
        }
    }
    // 4、清理未匹配/超龄的轨迹 (unmatched/old Tracks)
    vector<unique_ptr<Track>> active_tracks;
    vector<pair<int, BBox>> results;
    for (int i = 0; i < m_tracks.size(); ++i) {
        // 如果轨迹未超过最大丢失帧数 (max_age)
        if (m_tracks[i]->missed < m_sort_params.max_missed) {
            // 如果轨迹满足最小命中数 (min_hits)，则添加到结果中
            if (m_tracks[i]->hits >= m_sort_params.min_hits) {
                results.push_back({m_tracks[i]->id, m_tracks[i]->get_state_bbox()});
            }
            active_tracks.emplace_back(move(m_tracks[i]));  // 保留轨迹
        } else {
            // 轨迹丢失超过 max_age，将被删除
        }
    }

    // 更新轨迹列表
    m_tracks = move(active_tracks);

    return results;
}

/// @brief 跟踪器的预测和关联
/// @param detections 当前帧的边界框检测列表
/// @return 匹配之后的矩阵 (i,j)=1表示第i个预测框与第j个检测框匹配
vector<vector<double>> SortTracker::predict_and_associate(const vector<BBox> detections) {
    // 1、预测现有的所有轨迹
    std::vector<BBox> pred_boxes;
    for (auto& track : m_tracks) {
        track->predict();
        pred_boxes.push_back(track->get_state_bbox());
    }

    // 使用匈牙利算法进行匹配 lap指派任务求解
    auto matched_indices = solve_assignment(pred_boxes, detections, m_sort_params.iou_threshold);

    return matched_indices;
}

/// @brief 计算 跟踪器预测的bbox和检测的bbox的IOU矩阵，用于匹配计算
/// @param tarck_bboxs 跟踪器预测的框 [x1,y1,x2,y2]
/// @param det_bboxs 检测的bbox框 [x1,y1,x2,y2]
/// @return iou值
double SortTracker::calculate_iou(const BBox& rect1, const BBox& rect2) {
    float x_min_i = std::max(rect1(0), rect2(0));
    float y_min_i = std::max(rect1(1), rect2(1));
    float x_max_i = std::min(rect1(2), rect2(2));
    float y_max_i = std::min(rect1(3), rect2(3));

    float intersection_w = std::max(0.0f, x_max_i - x_min_i);
    float intersection_h = std::max(0.0f, y_max_i - y_min_i);
    float intersection_area = intersection_w * intersection_h;

    float area1 = (rect1(2) - rect1(0)) * (rect1(3) - rect1(1));
    float area2 = (rect2(2) - rect2(0)) * (rect2(3) - rect2(1));

    // 并集面积 = 面积1 + 面积2 - 交集面积
    float union_area = area1 + area2 - intersection_area;

    if (union_area <= 0.0f) return 0.0f;
    return intersection_area / union_area;
}

/// @brief 线性指派求解，使用lapjv求解
/// @param pred_boxes 预测的框
/// @param detections 检测的框
/// @param iou_threshold iou阈值
/// @return 匹配之后的矩阵 (i,j)=1表示第i个预测框与第j个检测框匹配
vector<vector<double>> SortTracker::solve_assignment(const vector<BBox>& pred_boxes, const vector<BBox>& detections,
                                       float iou_threshold) {
    int num_tracks = m_tracks.size();
    int num_dets = detections.size();
    int n = num_tracks >= num_dets ? num_tracks : num_dets;
    vector<vector<double>> matched_indices(num_tracks,vector<double>(num_dets,0));
    vector<vector<double>> iou_matrix(num_tracks,vector<double>(num_dets,0));

    // 如果没有轨迹或检测，直接返回空匹配
    if (num_tracks == 0 || num_dets == 0) {
        return matched_indices;
    }

    // 构建并初始化 cost 矩阵（填充值为 1.0，表示最大代价：iou = 0）
    std::vector<std::vector<double>> cost(n, std::vector<double>(n, 1.0));
    // 填写实际的代价与 iou 矩阵值
    for (int i = 0; i < num_tracks; ++i) {
        for (int j = 0; j < num_dets; ++j) {
            double iou = calculate_iou(pred_boxes[i], detections[j]);
            iou_matrix[i][j] = iou;
            cost[i][j] = 1.0 - iou;
        }
    }

    // 创建指向每一行的指针数组，lapjv_internal 期望的是 cost_t *cost[]
    std::vector<double*> cost_ptrs(n);
    for (int i = 0; i < n; ++i) cost_ptrs[i] = cost[i].data();

    std::vector<int> row_index(n), col_index(n);
    lapjv_internal(n, cost_ptrs.data(), row_index.data(), col_index.data());
    
    for (int i = 0; i < num_tracks; ++i) {
        int assigned_col = row_index[i];
        // 仅当分配的列在检测数量范围内，并且满足 iou 阈值时才认为是有效匹配
        if (assigned_col >= 0 && assigned_col < num_dets) {
            if (iou_matrix[i][assigned_col] >= iou_threshold) {  // 筛选满足阈值的分配结果
                matched_indices[i][assigned_col] = 1;
            }
        }
    }
    return matched_indices;
}
