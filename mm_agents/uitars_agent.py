# The following codes are based on OSWorld(https://github.com/xlang-ai/OSWorld/tree/main).
# For codes from mm_agents, please refer to the original OSWorld repository.

import base64
import json
import logging
import os
from io import BytesIO
from typing import Dict, List
import copy
import math
import numpy as np
from openai import OpenAI
from PIL import Image

from .action_parser import (
    parse_action_qwen2vl,
    parsing_response_to_pyautogui_code,
    FINISH_WORD,
    WAIT_WORD,
    ENV_FAIL_WORD,
    CALL_USER
)

logger = logging.getLogger("desktopenv.agent")

def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")  # 你可以改成 "JPEG" 等格式
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

class PromptAgent:
    def __init__(
        self,
        platform="ubuntu",
        max_tokens=1000,
        top_p=0.9,
        temperature=0.0,
        action_space="computer_13",
        observation_type="screenshot_a11y_tree",
        # observation_type can be in ["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"]
        max_trajectory_length=50,
        a11y_tree_max_tokens=10000,
        runtime_conf: dict = {}
    ):
        self.platform = platform
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length # same as max_steps
        self.a11y_tree_max_tokens = a11y_tree_max_tokens
        self.runtime_conf = runtime_conf
        self.vlm = OpenAI(
            base_url="http://8.130.84.63:55129/v1",
            api_key="empty",
        ) # should replace with your UI-TARS server api
        self.model = self.vlm.models.list().data[0].id
        self.input_swap = self.runtime_conf["input_swap"]
        self.language = self.runtime_conf["language"]
        self.max_steps = self.runtime_conf["max_steps"]

        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []
        
        self.customize_action_parser = parse_action_qwen2vl
        self.action_parse_res_factor = 1000
        if "history_n" in self.runtime_conf:
            self.history_n = self.runtime_conf["history_n"]
        else:
            self.history_n = 5

    def predict(self, instruction: str, obs: Dict) -> List:
        """
        Predict the next action(s) based on the current observation.
        """

        # Append trajectory
        # print(len(self.observations), len(self.actions), len(self.actions))
        # import pdb;pdb.set_trace()
        assert len(self.observations) == len(self.actions) and len(self.actions) == len(
            self.thoughts
        ), "The number of observations and actions should be the same."

        self.history_images.append(obs["screenshot"])

        if self.observation_type in ["screenshot", "screenshot_a11y_tree"]:
            base64_image = obs["screenshot"]
            self.observations.append(
                {"screenshot": base64_image, "accessibility_tree": None}
            )

        else:
            raise ValueError(
                "Invalid observation_type type: " + self.observation_type
            )  # 1}}}
        
        if len(self.history_images) > self.history_n:
            self.history_images = self.history_images[-self.history_n:]

        max_pixels = 1350 * 28 * 28
        min_pixels = 100 * 28 * 28
        messages, images = [], []
        if isinstance(self.history_images, bytes):
            self.history_images = [self.history_images]
        elif isinstance(self.history_images, np.ndarray):
            self.history_images = list(self.history_images)
        elif isinstance(self.history_images, list):
            pass
        else:
            raise TypeError(f"Unidentified images type: {type(self.history_images)}")
        max_image_nums_under_32k = int(32768*0.75/max_pixels*28*28)
        if len(self.history_images) > max_image_nums_under_32k:
            num_of_images = min(5, len(self.history_images))
            max_pixels = int(32768*0.75) // num_of_images

        for turn, image in enumerate(self.history_images):
            if len(images) >= 5:
                break
            try:
                image = Image.open(BytesIO(image))
            except Exception as e:
                raise RuntimeError(f"Error opening image: {e}")

            if image.width * image.height > max_pixels:
                """
                如果图片超过/低于像素限制，则计算一个缩放因子resize_factor，使图片的像素数缩小到等于或小于max_pixels。这个缩放因子是通过开平方根计算的，确保纵横比保持不变,这样原始的相对坐标可以不经转换直接复用
                """
                resize_factor = math.sqrt(max_pixels / (image.width * image.height))
                width, height = int(image.width * resize_factor), int(image.height * resize_factor)
                image = image.resize((width, height))
            if image.width * image.height < min_pixels:
                resize_factor = math.sqrt(min_pixels / (image.width * image.height))
                width, height = math.ceil(image.width * resize_factor), math.ceil(image.height * resize_factor)
                image = image.resize((width, height))

            if image.mode != "RGB":
                image = image.convert("RGB")

            images.append(image)

        # user_prompt = f"You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action(s) to complete the task. \n\nIf the elements involved in the upcoming actions are already visible on the current screen and their positions are not affected by prior actions, you may output a series of actions at once.\n\n## Output Format\n```\nThought: ...\nAction: ...\n```\n\n## Action Space\n{self.prompt_action_space}\n\n## Note\n- Use English in `Thought` part.\n- Summarize your next action (with its target element) in one sentence in `Thought` part.\nIf multiple actions are output, list them line by line under Action:.\n\n## User Instruction\n{instruction}\n"
        user_prompt = f"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

        ## Output Format
        ```
        Thought: ...
        Action: ...
        ```

        ## Action Space

        click(start_box='<|box_start|>(x1,y1)<|box_end|>')
        left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
        right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
        drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
        hotkey(key='')
        type(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content. 
        scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
        wait() #Sleep for 5s and take a screenshot to check for any changes.
        finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.
        call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.


        ## Note
        - Use English in `Thought` part.
        - Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

        ## User Instruction
        {instruction}
        """

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}]
            }
        ]
        
        image_num = 0
        if len(self.history_responses) > 0:
            for history_idx, history_response in enumerate(self.history_responses):
                # send at most history_n images to the model
                if history_idx + self.history_n > len(self.history_responses):

                    cur_image = images[image_num]
                    encoded_string = pil_to_base64(cur_image)
                    messages.append({
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}]
                    })
                    image_num += 1
                    
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": history_response}]
                })

            cur_image = images[image_num]
            encoded_string = pil_to_base64(cur_image)
            messages.append({
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}]
            })
            image_num += 1
        
        else:
            cur_image = images[image_num]
            encoded_string = pil_to_base64(cur_image)
            messages.append({
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}]
            })
            image_num += 1

        try_times = 3
        while True:
            if try_times <= 0:
                print(f"Reach max retry times to fetch response from client, as error flag.")
                return "client error", ["DONE"]
            try:
                
                response = self.vlm.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    frequency_penalty=1,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                prediction = response.choices[0].message.content.strip()
                print("[PromptAgent] Prediction: ", prediction)
                log_messages = copy.deepcopy(messages)
                image_num = len(self.actions)
                for msg in reversed(log_messages):
                    for c in msg["content"]:
                        if "image_url" in c.keys():
                            c["image_url"]["url"] = f"step_{image_num}.png"
                            image_num -= 1
                log_messages.append({"prediction": prediction})
                json.dump(log_messages, open(os.path.join(self.example_result_dir, f"message_{len(self.actions)}.json"), "w"), indent=2)
                
                break
            except Exception as e:
                print(f"Error when fetching response from client, with error: {e}")
                if self.error_logger is not None:
                    self.error_logger.error(f"Error when fetching response from client: {e}")
                prediction = None
                try_times -= 1
                
        if prediction is None:
            return "client error", ["DONE"]
        
        self.history_responses.append(prediction)
        self.thoughts.append(prediction)
        self.runtime_logger.info(f"[Step {len(self.actions) + 1}] {prediction}\n")
        try:
            parsed_responses = self.customize_action_parser(
                prediction,
                self.action_parse_res_factor,
                self.runtime_conf["screen_height"],
                self.runtime_conf["screen_width"]
            )
            # parsed_responses = parse_action_to_structure_output(
            #     prediction,
            #     self.action_parse_res_factor,
            #     origin_resized_height,
            #     origin_resized_width,
            #     "qwen25vl",
            #     MAX_PIXELS,
            #     MIN_PIXELS
            # )
        except Exception as e:
            print(f"Parsing action error: {prediction}, with error:\n{e}")
            if self.error_logger is not None:
                self.error_logger.error(f"Parsing action error: {prediction}, with error:\n{e}")
            import traceback
            traceback.print_exc()
            return f"Parsing action error: {prediction}, with error:\n{e}", ["DONE"]
        actions = []
        for parsed_response in parsed_responses:
            if "action_type" in parsed_response:

                if parsed_response["action_type"] == FINISH_WORD:
                    self.actions.append(actions)
                    return prediction, ["DONE"]
                
                elif parsed_response["action_type"] == WAIT_WORD:
                    self.actions.append(actions)
                    return prediction, ["WAIT"]
                
                elif parsed_response["action_type"] == ENV_FAIL_WORD:
                    self.actions.append(actions)
                    return prediction, ["FAIL"]

                elif parsed_response["action_type"] == CALL_USER:
                    self.actions.append(actions)
                    return prediction, ["FAIL"]
            
            pyautogui_code, _ = parsing_response_to_pyautogui_code(
                parsed_response,
                self.runtime_conf["screen_height"],
                self.runtime_conf["screen_width"],
                self.input_swap
            )
            self.runtime_logger.info(f"[Code] {pyautogui_code}\n")
            
            actions.append(pyautogui_code)

        self.actions.append(actions)
        if len(self.history_responses) >= self.max_trajectory_length:
            actions = ["FAIL"]

        return prediction, actions

    def reset(self, runtime_logger=None, error_logger=None, example_result_dir=None):
        self.runtime_logger = runtime_logger
        self.error_logger = error_logger
        self.example_result_dir = example_result_dir
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []