#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Kuavo 下位机桥接节点

- 订阅 /cmd_vel（线速度 + 角速度）
- 通过 UDP 将 {v, w, stamp} 发送给下位机

实际部署时，你可以根据 Kuavo 官方 SDK 协议修改 send_command 的实现。
"""

import json
import socket

import rospy
from geometry_msgs.msg import Twist


class KuavoLowlevelBridge(object):
    def __init__(self):
        self.ip = rospy.get_param('~kuavo_ip', '192.168.1.100')
        self.port = int(rospy.get_param('~kuavo_port', 9000))

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dest = (self.ip, self.port)

        rospy.Subscriber('/cmd_vel', Twist, self.cmd_cb, queue_size=1)
        rospy.loginfo('KuavoLowlevelBridge 目标 %s:%d', self.ip, self.port)

    def cmd_cb(self, msg):
        data = {
            'v': float(msg.linear.x),
            'w': float(msg.angular.z),
            'stamp': rospy.Time.now().to_sec(),
        }
        payload = json.dumps(data).encode('utf-8')
        try:
            self.sock.sendto(payload, self.dest)
        except Exception as e:
            rospy.logwarn('发送到 Kuavo 下位机失败: %s', e)


def main():
    rospy.init_node('kuavo_lowlevel_bridge_node')
    bridge = KuavoLowlevelBridge()
    rospy.loginfo('kuavo_lowlevel_bridge_node 启动完成')
    rospy.spin()


if __name__ == '__main__':
    main()
