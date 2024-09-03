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

from rclpy.node import Node
from rclpy.action import ActionClient

from whisper_msgs.action import STT

from llama_ros.langchain.llama_ros import LlamaClientNode
from llama_msgs.action import GenerateResponse

from sttb_msgs.action import MoveRobot


class STTBNode(Node):

    def __init__(self) -> None:
        super().__init__("sttb_node")

        # self.prompt = "You are an assistant that parses the user input and generates a response with the direction and the distance to move the robot.\n"
        self.prompt = "Eres un asistente que analiza la entrada del usuario y genera una lista de json de respuesta con un objeto que tiene como clave valor, la hacia donde se tiene que mover y los metros que se tiene que mover el robot. Devuelve la lista en el orden cronologico que el usuario quiere. "

        # self.MODEL_NAME = 'Meta-Llama-3.1-8B-Instruct-Q4_K_M'
        # self.MODEL_NAME = 'Phi-3.5-mini-instruct-Q4_K_M'
        # self.MODEL_NAME = 'Spaetzle-v60-7b-q4-k-m'
        # self.MODEL_NAME = 'gemma-2-9b-it-Q4_K_M'
        self.MODEL_NAME = 'internlm2_5-7b-chat-q4_k_m'

        self.tokens = 0
        self.initial_time = -1
        self.eval_time = -1
        self.llm_response = ''

        self.accumulative_time = 0
        self.accumulative_tokens_per_second = 0
        self.accumulative_tokens = 0
        self.accumulative_matches = 0
        self.accumulative_misses = 0

        self.goal = GenerateResponse.Goal()
        self.goal.sampling_config.temp = 0.2
        self.goal.sampling_config.penalty_last_n = 0
        self.goal.sampling_config.n_prev = 0
        self.goal.reset = True
        self.goal.sampling_config.grammar = open('../grammars/sttb_array_grammar.gbnf').read()

        self._whisper_client = ActionClient(self, STT, "/whisper/listen")
        self._llama_client = LlamaClientNode.get_instance("llama")

    def text_cb(self, feedback) -> None:
        if self.eval_time < 0:
            self.eval_time = time.time()

        self.tokens += 1
        # print(feedback.feedback.partial_response.text, end="", flush=True)
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

    def llm_parse(self, whisper_result: str, expected_text: str) -> None:


        self.goal.prompt = self.prompt + whisper_result

        # Reset of the metrics
        self.llm_response = ''
        self.tokens = 0
        self.eval_time = -1

        # Start the timer
        self.initial_time = time.time()

        self._llama_client.generate_response(self.goal, self.text_cb)

        # End the timer
        end_time = time.time()

        # Clean the response to compare it
        llm_response_clean = self.llm_response.replace(' ', '').replace('\n', '').replace('\t', '')

        # Calculate the individual metrics
        exact_content = llm_response_clean==expected_text.replace(' ', '').replace('\n', '').replace('\t', '')
        time_diff = round(self.eval_time - self.initial_time, 4)
        tokens_per_second = round(self.tokens / (end_time - self.eval_time), 4)

        self.get_logger().info(
            f"Time: {self.eval_time - self.initial_time} s, speed: {self.tokens / (end_time - self.eval_time)} t/s")
        self.get_logger().info(llm_response_clean)

        # Calculate the accumulative metrics
        if exact_content:
            self.accumulative_matches += 1
        else:
            self.accumulative_misses += 1

        self.accumulative_time += time_diff
        self.accumulative_tokens_per_second += tokens_per_second
        self.accumulative_tokens += self.tokens

        # Add the results to the results.csv file
        with open(f'../results/results.csv', 'a') as f:
            f.write(f"{self.MODEL_NAME};{whisper_result};{time_diff};{self.tokens};{tokens_per_second};{exact_content};{llm_response_clean}\n")


    def main_flow (self) -> None:

        # whisper_result = ''

        # whisper_result = self.listen()

        n_tests = 50

        result_list = [
            {
                "text": "Desplazate 3 metros a la izquierda y luego otros 3 metros hacia adelante",
                "expected_result": "[{'direction': 'LEFT', 'distance': 3.00}, {'direction': 'FRONT', 'distance': 3.00}]"
            },
            {
                "text": "Muevete 2 metros a la derecha, pero antes 1 metro atras pero lo primero de todo muevete 3 metros a la izquierda",
                "expected_result": "[{'direction': 'LEFT', 'distance': 3.00}, {'direction': 'BACK', 'distance': 1.00}, {'direction': 'RIGHT', 'distance': 2.00}]"
            },
            {
                "text": "En segundo lugar avanza hacia adelante 1 metro, en tercer lugar retrocede 2 metros y en primer lugar muevete medio metro hacia la derecha",
                "expected_result": "[{'direction': 'RIGHT', 'distance': 0.50}, {'direction': 'FRONT', 'distance': 1.00}, {'direction': 'BACK', 'distance': 2.00}]"
            },
            {
                "text": "Muevete 3,47 metros a la izquierda, luego 1 metro y 78 centimetros hacia adelante, y por ultimo, 30 centimetros hacia la derecha",
                "expected_result": "[{'direction': 'LEFT', 'distance': 3.47}, {'direction': 'FRONT', 'distance': 1.78}, {'direction': 'RIGHT', 'distance': 0.30}]"
             },
            {
                "text": "¿Seria tan amable de moverse 3 metros a la derecha, dos metros hacia adelante y luego deshacer todos sus pasos?",
                "expected_result": "[{'direction': 'RIGHT', 'distance': 3.00}, {'direction': 'FRONT', 'distance': 2.00}, {'direction': 'BACK', 'distance': 2.00},{'direction': 'LEFT', 'distance': 3.00}]"
             },
        ]

        for result in result_list:
            for i in range(n_tests):
                whisper_result = result.get('text')

                expected_text = result.get('expected_result')

                self.llm_parse(whisper_result, expected_text)

            with open(f'../results/avg_results.csv', 'a') as f:
                f.write(
                    f"{self.MODEL_NAME};{result.get('text')};{round(self.accumulative_time/n_tests, 4)};"
                    f"{round(self.accumulative_tokens/n_tests, 4)};{round(self.accumulative_tokens_per_second/n_tests ,4)};"
                    f"{round(self.accumulative_matches / (self.accumulative_matches + self.accumulative_misses), 4)};"
                    f"{self.accumulative_matches};{self.accumulative_misses}\n"
                )

            # Reset the accumulative metrics
            self.accumulative_time = 0
            self.accumulative_tokens_per_second = 0
            self.accumulative_tokens = 0
            self.accumulative_matches = 0
            self.accumulative_misses = 0

    # def test_method(self):
    #     goal = MoveRobot.Goal()
    #     goal.x = 3.0
    #     goal.y = 0.0
    #
    #     self._move_robot_action_client.wait_for_server()
    #
    #     send_goal_future = self._move_robot_action_client.send_goal(goal)


def main():
    rclpy.init()
    node = STTBNode()
    node.main_flow()
    # node.test_method()
    rclpy.shutdown()


if __name__ == "__main__":
    main()