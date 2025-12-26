
#include <KalmanFilter.h>

/// @brief 初始化固定参数 减少后面目标实例化的时候多次赋值
/// @param state_dim 状态维度
/// @param meas_dim 测量维度
/// @param params 卡尔曼滤波器的参数
void KalmanFilter::initGlobalParams(const KalmanFilterParam &params) {
    // 状态转换矩阵 F: 采用恒速模型 x' = Fx
    F = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
    F.block<4, 4>(0, 4) = Eigen::Matrix4d::Identity();
    // F(0, 4) = F(1, 5) = F(2, 6) = F(3, 7) = 1.0;

    // 测量矩阵 H: 只测量前4个状态 (位置和尺寸) z = H*x: [cx, cy, s, r] = [cx, cy, s, r, vx, vy, vs, vr]
    H = Eigen::MatrixXd::Zero(MEAS_DIM, STATE_DIM);
    H(0, 0) = H(1, 1) = H(2, 2) = H(3, 3) = 1.0;

    // 过程噪声协方差矩阵 Q: 恒速模型下的随机加速度扰动
    Q = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
    // 位置/尺寸噪声
    Q.block<4, 4>(0, 0) = Eigen::Matrix4d::Identity() * (1 * params.std_dev_process * params.std_dev_process);
    Q.block<4, 4>(4, 4) = Eigen::Matrix4d::Identity() * (0.01* params.std_dev_process * params.std_dev_process);  // 速度噪声

    // 测量噪声协方差矩阵 R
    R = Eigen::MatrixXd::Identity(MEAS_DIM, MEAS_DIM) * (params.std_dev_measure * params.std_dev_measure);
}

/// @brief 根据检测结果初始化
/// @param bbox 检测BoudingBox [x1,y1,x2,y2]
KalmanFilter::KalmanFilter(const Eigen::Vector4d &bbox) {
    // 初始化状态和协方差矩阵
    x.template head<4>() = Bbox2KalmanState(bbox);  // 前四个进行赋值
    P = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) * m_params.initial_P_value;
}

/**
 * @brief 预测下一步状态和协方差
 */
void KalmanFilter::predict() {
    // 预测状态: x'_k = F * x_{k-1}
    x = F * x;
    // 预测协方差矩阵: P'_k = F * P_{k-1} * F^T + Q
    P = F * P * F.transpose() + Q;
}
/**
 * @brief 使用测量值更新状态和协方差。
 * @param bbox : [x1, y1, x2, y2]
 */
void KalmanFilter::update(const Eigen::Vector4d &bbox) {
    Eigen::VectorXd z = Bbox2KalmanState(bbox);
    // 计算卡尔曼增益 K
    Eigen::Matrix4d S = H * P * H.transpose() + R;                    // 测量残差协方差 S = H * P'_k * H^T + R
    Eigen::Matrix<double, 8, 4> K = P * H.transpose() * S.inverse();  // 卡尔曼增益 K = P'_k * H^T * S^{-1}

    // 更新状态 x_k = x'_k + K * (z_k - H * x'_k)
    x = x + K * (z - H * x);

    // 更新协方差 P_k = (I - K * H) * P'_k
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
    P = (I - K * H) * P;
}

Eigen::Vector4d KalmanFilter::Bbox2KalmanState(const Eigen::Vector4d &bbox) {
    Eigen::Vector4d z;
    double cx = (bbox(0) + bbox(2)) / 2;
    double cy = (bbox(1) + bbox(3)) / 2;
    double s = (bbox(2) - bbox(0)) * (bbox(3) - bbox(1));
    double r = (bbox(2) - bbox(0)) / (bbox(3) - bbox(1));
    z << cx, cy, s, r;
    return z;
}

Eigen::Vector4d KalmanFilter::KalmanState2Bbox() {
    Eigen::Vector4d bbox;
    if (x(2) < 0 || x(3) < 0) {  // 确保面积和框高比为非负，满足物理约束
        bbox << 0.0, 0.0, 0.0, 0.0;
    } else {
        double w = sqrt(x(2) * x(3));
        double h = sqrt(x(2) / x(3));
        double x1 = x(0) - w / 2;
        double y1 = x(1) - h / 2;
        double x2 = x(0) + w / 2;
        double y2 = x(1) + h / 2;
        bbox << x1, y1, x2, y2;
    }
    return bbox;
}
