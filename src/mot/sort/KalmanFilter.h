// KalmanFilter 头文件

#ifndef __KALMANFILTER_H__
#define __KALMANFILTER_H__

#include <Eigen/Dense>
#include <cmath>

typedef Eigen::Matrix<double,8,1> Vector8d;
typedef Eigen::Matrix<double,8,8> Matrix8d;

#define STATE_DIM 8  // 状态维度 [c_x, c_y, s, r, vx, vy, vs, vr]^T
#define MEAS_DIM 4   // 测量维度 [c_x, c_y, s, r]^T 检测框的中心点、面积和宽高比

/**
 * @brief 卡尔曼滤波的配置参数。
 */
struct KalmanFilterParam {
    double std_dev_process;  // 过程噪声的标准差（用于Q矩阵）
    double std_dev_measure;  // 测量噪声的标准差（用于R矩阵）
    double initial_P_value;  // 初始协方差矩阵对角线值
};

class KalmanFilter {
private:
    // 需要更新的变量
    Vector8d x = Vector8d::Zero();  // 当前状态 (8x1)
    Matrix8d P = Matrix8d::Zero();  // 状态协方差矩阵 (8x8)

    // 转换矩阵 (F), 测量矩阵 (H), 过程噪声协方差 (Q), 测量噪声协方差 (R)
    static inline Eigen::MatrixXd F;  // 状态转换矩阵 (8x8)
    static inline Eigen::MatrixXd H;  // 测量矩阵 (4x8)
    static inline Eigen::MatrixXd Q;  // 过程噪声协方差矩阵 (8x8)
    static inline Eigen::MatrixXd R;  // 测量噪声协方差矩阵 (4x4)

    // 卡尔曼滤波器参数
    KalmanFilterParam m_params{};

public:
    static void initGlobalParams(const KalmanFilterParam &params);      // 初始化静态 F H Q R
    KalmanFilter(const KalmanFilterParam &params) : m_params(params) {  // 初始化参数
        initGlobalParams(params);
    };
    KalmanFilter(const Eigen::Vector4d &bbox);               // 通过检测框初始化变量
    void predict();                                   // 预测
    void update(const Eigen::Vector4d &z);                   // 更新
    Eigen::Vector4d Bbox2KalmanState(const Eigen::Vector4d &bbox);  // 将检测框[x1,y1,x2,y2]转换成kalman状态变量[cx,cy,s,r]
    Eigen::Vector4d KalmanState2Bbox();                      // 将kalman状态变量[cx,cy,s,r]转换成检测框[x1,y1,x2,y2]
    ~KalmanFilter() = default;
};

#endif  // __KALMANFILTER_H__