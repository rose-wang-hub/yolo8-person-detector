#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""基于 PersonState 的跟随控制节点

- 订阅 /person_track/vision_state（或未来的 /person_track/fused_state）
- 使用简单 PID 计算期望线速度 v 和角速度 w
- 发布到 /cmd_vel，供下位机或仿真使用
"""

import time

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from kuavo_person_follow.msg import PersonState


class PIDFollower(object):
    def __init__(self):
        self.dist_target = rospy.get_param('~dist_target', 1.5)

        self.kp_dist = rospy.get_param('~kp_dist', 0.8)
        self.ki_dist = rospy.get_param('~ki_dist', 0.0)
        self.kd_dist = rospy.get_param('~kd_dist', 0.1)

        self.kp_yaw = rospy.get_param('~kp_yaw', 1.2)
        self.ki_yaw = rospy.get_param('~ki_yaw', 0.0)
        self.kd_yaw = rospy.get_param('~kd_yaw', 0.1)

        self.v_max = rospy.get_param('~v_max', 0.8)
        self.yaw_rate_max = rospy.get_param('~yaw_rate_max', 1.0)

        self._int_dist = 0.0
        self._prev_dist_err = 0.0
        self._int_yaw = 0.0
        self._prev_yaw_err = 0.0
        self._last_time = None

    def compute(self, state):
        now = time.time()
        if self._last_time is None:
            dt = 0.01
        else:
            dt = max(1e-3, now - self._last_time)
        self._last_time = now

        if not state.valid:
            return 0.0, 0.0

        dist_err = state.z - self.dist_target
        self._int_dist += dist_err * dt
        d_dist = (dist_err - self._prev_dist_err) / dt
        self._prev_dist_err = dist_err

        v_cmd = (self.kp_dist * dist_err +
                 self.ki_dist * self._int_dist +
                 self.kd_dist * d_dist)

        yaw_err = state.yaw_err
        self._int_yaw += yaw_err * dt
        d_yaw = (yaw_err - self._prev_yaw_err) / dt
        self._prev_yaw_err = yaw_err

        yaw_rate_cmd = (self.kp_yaw * yaw_err +
                        self.ki_yaw * self._int_yaw +
                        self.kd_yaw * d_yaw)

        v_cmd = max(-self.v_max, min(self.v_max, v_cmd))
        yaw_rate_cmd = max(-self.yaw_rate_max,
                           min(self.yaw_rate_max, yaw_rate_cmd))
        return v_cmd, yaw_rate_cmd


class FollowControllerNode(object):
    def __init__(self):
        self.pid = PIDFollower()

        # 允许通过参数指定输出控制话题，默认沿用通用的 /cmd_vel
        # 在 Kuavo4Pro 上建议在 launch 中将其设为 /cmd_vel_app，
        # 通过现有的多路复用/安全模块再下发到下位机。
        cmd_topic = rospy.get_param('~cmd_topic', '/cmd_vel')
        self.pub = rospy.Publisher(cmd_topic, Twist, queue_size=10)

        topic = rospy.get_param('~state_topic', '/person_track/vision_state')
        rospy.Subscriber(topic, PersonState, self.state_cb, queue_size=1)
        rospy.loginfo('follow_controller_node 订阅 %s', topic)

        # 跟随状态超时保护：若长时间没有新的 PersonState，则自动发布 0 速度
        self.state_timeout = rospy.get_param('~state_timeout', 0.5)
        self._last_state_time = None
        self._watchdog_timer = rospy.Timer(rospy.Duration(0.1), self._watchdog_cb)

        # 激光雷达安全控制参数
        self.scan_topic = rospy.get_param('~scan_topic', '/scan')
        self.front_angle_deg = rospy.get_param('~front_angle_deg', 30.0)
        self.stop_dist = rospy.get_param('~stop_dist', 0.6)
        self.slow_dist = rospy.get_param('~slow_dist', 1.0)
        self.scan_timeout = rospy.get_param('~scan_timeout', 0.5)

        self._min_front_dist = None
        self._last_scan_time = None

        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_cb, queue_size=1)
        rospy.loginfo('follow_controller_node 使用激光雷达 %s 做前向安全约束', self.scan_topic)

    def scan_cb(self, scan):
        """从 LaserScan 中计算前方扇区的最近距离，用于速度限幅。"""
        if not scan.ranges:
            return

        front_angle_rad = abs(self.front_angle_deg) * 3.141592653589793 / 180.0
        angles = []
        ranges = []
        angle = scan.angle_min
        for r in scan.ranges:
            angles.append(angle)
            ranges.append(r)
            angle += scan.angle_increment

        candidates = []
        for a, r in zip(angles, ranges):
            if abs(a) <= front_angle_rad:
                if r >= scan.range_min and r <= scan.range_max:
                    candidates.append(r)

        if not candidates:
            return

        self._min_front_dist = min(candidates)
        self._last_scan_time = rospy.Time.now().to_sec()

    def state_cb(self, state):
        v, w = self.pid.compute(state)

        # 结合激光雷达最近距离，对前向线速度做安全限幅
        v_safe = v
        now = rospy.Time.now().to_sec()
        if self._min_front_dist is not None and self._last_scan_time is not None:
            if now - self._last_scan_time < self.scan_timeout:
                d = self._min_front_dist
                if d < self.stop_dist:
                    # 前方过近，强制停车
                    v_safe = 0.0
                elif d < self.slow_dist and v_safe > 0.0:
                    # 在减速带内，按比例降低速度
                    ratio = (d - self.stop_dist) / max(1e-3, self.slow_dist - self.stop_dist)
                    ratio = max(0.0, min(1.0, ratio))
                    v_safe = v_safe * ratio

        cmd = Twist()
        cmd.linear.x = v_safe
        cmd.angular.z = w
        self.pub.publish(cmd)

        # 记录最近一次根据视觉状态发布控制命令的时间
        self._last_state_time = rospy.Time.now().to_sec()

    def _watchdog_cb(self, event):
        """看门狗：如果长时间没有新的 PersonState，则定期发布 0 速度。

        注意：Kuavo 下位机会维持最近一次非零 /cmd_vel 的效果，因此
        不能依赖 "不再发送" 来停车，必须主动发送 0 速度。
        """
        if self._last_state_time is None:
            return

        now = rospy.Time.now().to_sec()
        if now - self._last_state_time > self.state_timeout:
            cmd = Twist()
            # 默认各分量为 0，即线速度和角速度均为 0
            self.pub.publish(cmd)


def main():
    rospy.init_node('follow_controller_node')
    node = FollowControllerNode()
    rospy.loginfo('follow_controller_node 启动完成')
    rospy.spin()


if __name__ == '__main__':
    main()
