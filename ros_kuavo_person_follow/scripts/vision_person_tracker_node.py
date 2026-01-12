#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""视觉人体跟踪 ROS 节点

功能：
- 使用你现有的 HumanStateEstimator（RealSense + YOLO + ROI 选择）
- 将跟踪结果封装成 PersonState 消息并发布到 /person_track/vision_state

使用方式：
    rosrun kuavo_person_follow vision_person_tracker_node.py \
        _detector_root:=/home/eric/yolo8-person-detector

注意：
- 节点启动后会弹出 ROI 选择窗口，按 s/空格/回车确认后开始持续发布状态。
"""

import os
import sys

import rospy
from kuavo_person_follow.msg import PersonState


def import_human_estimator(detector_root):
    """将 yolo8-person-detector 目录加入 sys.path 并导入 HumanStateEstimator。"""
    detector_root = os.path.abspath(detector_root)
    if detector_root not in sys.path:
        sys.path.insert(0, detector_root)

    try:
        from kuavo_follow_controller import HumanStateEstimator, HumanState  # type: ignore
    except ImportError as e:
        rospy.logerr("无法从 %s 导入 kuavo_follow_controller: %s", detector_root, e)
        raise
    return HumanStateEstimator, HumanState


def main():
    rospy.init_node('vision_person_tracker_node')

    detector_root = rospy.get_param('~detector_root', '/home/eric/yolo8-person-detector')
    HumanStateEstimator, HumanState = import_human_estimator(detector_root)

    pub = rospy.Publisher('/person_track/vision_state', PersonState, queue_size=10)

    rate_hz = rospy.get_param('~rate', 30.0)
    rate = rospy.Rate(rate_hz)

    rospy.loginfo("vision_person_tracker_node 启动，detector_root=%s, rate=%.1f Hz",
                  detector_root, rate_hz)

    estimator = HumanStateEstimator()

    try:
        while not rospy.is_shutdown():
            state = estimator.get_state()  # HumanState

            msg = PersonState()
            msg.x = state.x
            msg.z = state.z
            msg.vx = state.vx
            msg.vz = state.vz
            msg.yaw_err = state.yaw_err
            msg.valid = bool(state.valid)
            msg.confidence = 1.0 if state.valid else 0.0

            pub.publish(msg)
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    finally:
        try:
            estimator.stop()
        except Exception:
            pass
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == '__main__':
    main()
