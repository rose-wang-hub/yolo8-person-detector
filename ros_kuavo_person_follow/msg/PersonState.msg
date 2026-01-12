float32 x        # 左右偏移 (m)，左为正（base_link 坐标系）
float32 z        # 前后距离 (m)，前方为正
float32 vx       # x 方向速度 (m/s)
float32 vz       # z 方向速度 (m/s)
float32 yaw_err  # 朝向误差 (rad)，>0 目标在左侧
bool   valid     # 当前是否有可靠目标
float32 confidence  # 0~1, 置信度
