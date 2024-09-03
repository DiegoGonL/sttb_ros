#!/usr/bin/env python3
import json
from typing import Optional

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
import time

from mercurial.templatefuncs import dict_
from rclpy.node import Node
from rclpy.action import ActionClient
from ros2action.verb.send_goal import send_goal
from torch.ao.quantization.fx import convert

from whisper_msgs.action import STT

from llama_ros.langchain.llama_ros import LlamaClientNode
from llama_msgs.action import GenerateResponse

from sttb_msgs.action import MoveRobot
from sttb_msgs.msg import MoveRobot as MoveRobotMsg


class STTBNode(Node):

    def __init__(self) -> None:
        super().__init__("sttb_node")

        # self.prompt = "You are an assistant that parses the user input and generates a response with the direction and the distance to move the robot.\n"
        self.prompt = "Eres un asistente que analiza la entrada del usuario y genera una lista de json de respuesta con un objeto que tiene como clave valor, la hacia donde se tiene que mover y los metros que se tiene que mover el robot. Devuelve la lista en el orden cronologico que el usuario quiere. "

        self.tokens = 0
        self.confidence = None
        self.initial_time = -1
        self.eval_time = -1
        self.llm_response = ''

        self._whisper_client = ActionClient(self, STT, "/whisper/listen")
        self._llama_client = LlamaClientNode.get_instance("llama")

        self._move_robot_action_client = ActionClient(self, MoveRobot, '/sttb/move_robot')
        self._move_robot_publisher = self.create_publisher(MoveRobotMsg, '/sttb/move_robot', 10)

        self._eval_array = []

    def text_cb(self, feedback) -> None:
        if self.eval_time < 0:
            self.eval_time = time.time()

        self.tokens += 1
        print(feedback.feedback.partial_response.text, end="", flush=True)
        self.llm_response += feedback.feedback.partial_response.text

    def listen(self) -> str:
        self.get_logger().info("Listening...")
        self._whisper_client.wait_for_server()
        goal = STT.Goal()
        send_goal_future = self._whisper_client.send_goal_async(goal)

        rclpy.spin_until_future_complete(self, send_goal_future)
        get_result_future = send_goal_future.result().get_result_async()
        self.get_logger().info("SPEAK")

        rclpy.spin_until_future_complete(self, get_result_future)
        result: STT.Result = get_result_future.result().result
        self.get_logger().info(f"I hear: {result.transcription.text}")
        self.get_logger().info(f"Audio time: {result.transcription.audio_time}")
        self.get_logger().info(f"Transcription time: {result.transcription.transcription_time}")

        return result.transcription.text

    def llm_parse(self, whisper_result: str) -> dict:
        goal = GenerateResponse.Goal()
        goal.prompt = self.prompt + whisper_result
        goal.sampling_config.temp = 0.2
        goal.sampling_config.penalty_last_n = 0
        goal.sampling_config.n_prev = 0
        goal.reset = True
        goal.sampling_config.grammar = open('../grammars/sttb_array_grammar.gbnf').read()

        self.llm_response = ''
        self.tokens = 0
        self._llama_client.generate_response(goal, self.text_cb)

        # Convert the llm_response to a list of dictionaries
        dict_list = json.loads(self.llm_response.replace("'", "\""))

        return dict_list

    def main_flow (self) -> None:

        whisper_result = ''

        while whisper_result.lower() not in ['exit.', 'exit', 'quit', 'quit.', 'salir', 'salir.']:

            whisper_result = self.listen()

            # Evaluate how much time it took to get the response
            self.initial_time = time.time()

            llm_result = self.llm_parse(whisper_result)

            self.eval_time = time.time() - self.initial_time

            print(f"Time to evaluate: {self.eval_time}, Tokens: {self.tokens}")
            print(llm_result)


            goal = MoveRobotMsg()

            for step in llm_result:
                if step.get('direction') == 'FRONT':
                    goal.x = step.get('distance')
                    goal.y = 0.0
                elif step.get('direction') == 'BACK':
                    goal.x = - step.get('distance')
                    goal.y = 0.0
                elif step.get('direction') == 'LEFT':
                    goal.x = 0.0
                    goal.y = step.get('distance')
                elif step.get('direction') == 'RIGHT':
                    goal.x = 0.0
                    goal.y = - step.get('distance')

                self._move_robot_publisher.publish(goal)

def main():
    rclpy.init()
    node = STTBNode()
    node.main_flow()
    rclpy.shutdown()


if __name__ == "__main__":
    main()