#!/usr/bin/env python3
from math import sqrt, atan2

# MIT License

# Copyright (c) 2023  Diego González López

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import rclpy

from nav_msgs.msg import Odometry

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from transformations import euler_from_quaternion

from geometry_msgs.msg import Twist, Pose

from sttb_msgs.action import MoveRobot
from sttb_msgs.msg import MoveRobot as MoveRobotMsg


class STTBMovementActionServer(Node):

    def __init__(self) -> None:
        super().__init__("sttb_movement_node")

        self.roll = None
        self.rate = self.create_rate(5)
        self.theta = None
        self.pose = Pose()

        self.goal_pose_queue = []

        as_cb_group = None
        odom_cb_group = ReentrantCallbackGroup()

        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.update_pose_cb, 10, callback_group=odom_cb_group)
        self.move_robot_subscriber = self.create_subscription(MoveRobotMsg, '/sttb/move_robot', self.move_robot_subs, 10, callback_group=None)

        # self._action_server = ActionServer(self, MoveRobot, "/sttb/move_robot", self.move_robot_cb, callback_group=as_cb_group)

    def update_pose_cb(self, data: Odometry) -> None:
        self.pose = data.pose.pose
        self.pose.position.x = round(self.pose.position.x, 2)
        self.pose.position.y = round(self.pose.position.y, 2)
        (self.roll, pitch, self.theta) = euler_from_quaternion([self.pose.orientation.w, self.pose.orientation.x, self.pose.orientation.y,
                                                           self.pose.orientation.z])
        # print(self.roll, pitch, self.theta)

    def euclidean_distance(self, goal_pose: Pose) -> float:
        return sqrt(pow((goal_pose.position.x - self.pose.position.x), 2) +
                    pow((goal_pose.position.y - self.pose.position.y), 2))

    def move2goal(self, goal_pose: Pose):
        vel_msg = Twist()

        for goal_pose in self.goal_pose_queue:

            goal_pose.position.x = self.pose.position.x + goal_pose.position.x
            goal_pose.position.y = self.pose.position.y + goal_pose.position.y

            while self.euclidean_distance(goal_pose) >= 0.2:
                artan = atan2(goal_pose.position.y - self.pose.position.y, goal_pose.position.x - self.pose.position.x) - self.theta
                print(artan)
                if artan >= 0.3:
                    vel_msg.angular.z = 0.25
                    vel_msg.linear.x = 0.0
                elif artan <= -0.3:
                    vel_msg.angular.z = 0.25
                    vel_msg.linear.x = 0.0
                else:
                    vel_msg.angular.z = 0.0
                    vel_msg.linear.x = 0.25

                self.velocity_publisher.publish(vel_msg)
                self.rate.sleep()

            vel_msg.linear.x = 0.0
            vel_msg.angular.z = 0.0
            self.velocity_publisher.publish(vel_msg)
            print(f"Destination reached {self.goal_pose_queue.pop(0)}")

    def move_robot_subs(self, msg: MoveRobotMsg) -> None:
        goal_pose = Pose()
        goal_pose.position.x = msg.x
        goal_pose.position.y = msg.y

        self.goal_pose_queue.append(goal_pose)
        if len(self.goal_pose_queue) == 1:
            self.move2goal(goal_pose)

    def move_robot_cb(self, goal_handle):
        self.get_logger().info(f"Received goal: {goal_handle.request}")
        goal_pose = Pose()
        goal_pose.position.x = self.pose.position.x + goal_handle.request.x
        goal_pose.position.y = self.pose.position.y + goal_handle.request.y

        self.goal_pose_queue.append(goal_pose)

        self.move2goal(goal_pose)
        goal_handle.succeed()
        result = MoveRobot.Result()
        self._action_server.notify_goal_done()
        result.finished = True
        return result

def main():
    rclpy.init()
    sttb_action_server = STTBMovementActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(sttb_action_server)
    executor.spin()
    # rclpy.spin(sttb_action_server)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
