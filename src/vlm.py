target_match_background = "Task: Determine if the object in each of the candidate images 1-9 is part of the *articulated* object(s) in the target image. Note that the articulated object may be in *different poses* or joint configurations across images. Output the results in JSON format."
target_match_background_fp = "Task: Determine if the object in each of the candidate images 1-9 is part of the *articulated* object(s) or the red or blue box in the target image. Note that the articulated object may be in *different poses* or joint configurations across images. If the candidate images 1-9 contains the red or blue box, please return true. If the candidate images 1-9 contains the articulated object(s), please return true. If the candidate images 1-9 contains neither the red or blue box nor the articulated object(s), please return false. Output the results in JSON format."
target_match_background_door = "Task: Determine if the object in each of the candidate images 1-9 is part of the *articulated* object(s) or the door in the target image. Note that the articulated object may be in *different poses* or joint configurations across images. If the candidate images 1-9 contains the door, please return true. If the candidate images 1-9 contains the articulated object(s), please return true. If the candidate images 1-9 contains neither the door nor the articulated object(s), please return false. Output the results in JSON format."

target_match_description = """Please return the results in JSON format as an array of objects. **The order of objects in the array must correspond to the order of candidate images provided (from 1 to 9).** Each object should contain the following fields:

- "image_id": Candidate image number (1-9)
- "is_same_object": Boolean value (true if same object, false if different)

JSON Output Template:

```json
[
  { "image_id": "1", "is_same_object": boolean },
  { "image_id": "2", "is_same_object": boolean },
  { "image_id": "3", "is_same_object": boolean },
  { "image_id": "4", "is_same_object": boolean },
  { "image_id": "5", "is_same_object": boolean },
  { "image_id": "6", "is_same_object": boolean },
  { "image_id": "7", "is_same_object": boolean },
  { "image_id": "8", "is_same_object": boolean },
  { "image_id": "9", "is_same_object": boolean }
]
```
"""

import os
import base64
import io
import time
import json
import numpy as np
import torch
from PIL import Image
from openai import OpenAI

class VLM_Model(object):
    def __init__(self, args, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model_name="qwen-vl-max"):
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if api_key is None:
            raise ValueError("Please set the environment variable DASHSCOPE_API_KEY")
        self.args = args
        self.domain_name = args.domain_name
        self.task_name = args.task_name
        self.channels = args.channels
        self.region_num = args.masked_region_num
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        if self.domain_name == "franka":
            if self.task_name =="push":
                self.original = Image.open(f"./src/env/data/franka_push.png")
            else:
                self.original = Image.open(f"./src/env/data/franka.png")
        elif self.domain_name == 'robosuite':
            if self.task_name == "Door":
                self.original = Image.open(f"./src/env/data/door.png")
            else:
                raise NotImplementedError(f"Domain {self.domain_name} and task {self.task_name} not implemented")
        else:
            self.original = Image.open(f"./src/env/data/{self.domain_name}.png")
        self.original_base64 = self.img_to_base64(self.original)

    def arr_to_img(self, arr):
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        arr = arr.astype(np.uint8)
        return Image.fromarray(arr, mode='RGB')

    def img_to_base64(self, img):
        img_byte_array = io.BytesIO()
        img.save(img_byte_array, format='PNG')
        img_base64 = base64.b64encode(img_byte_array.getvalue()).decode("utf-8")
        return img_base64
    
    def chat_with_client(self, fragments, max_retries=5, delay=10):
        """
        fragments: List[Image]
        """
        origin_base64 = self.original_base64
        base64_images = [self.img_to_base64(fragment) for fragment in fragments]
        if self.domain_name == "franka" and self.task_name == "push":
            bg = target_match_background_fp
        elif self.domain_name == "robosuite" and self.task_name == "Door":
            bg = target_match_background_door
        else:
            bg = target_match_background
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": bg},
                                {"type": "text", "text": "Target Image:"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{origin_base64}"}, # base64_image
                                },
                                {"type": "text", "text": "Candidate Image 1:"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_images[0]}"},
                                },
                                {"type": "text", "text": "Candidate Image 2:"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_images[1]}"},
                                },
                                {"type": "text", "text": "Candidate Image 3:"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_images[2]}"},
                                },
                                {"type": "text", "text": "Candidate Image 4:"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_images[3]}"},
                                },
                                {"type": "text", "text": "Candidate Image 5:"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_images[4]}"},
                                },
                                {"type": "text", "text": "Candidate Image 6:"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_images[5]}"},
                                },
                                {"type": "text", "text": "Candidate Image 7:"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_images[6]}"},
                                },
                                {"type": "text", "text": "Candidate Image 8:"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_images[7]}"},
                                },
                                {"type": "text", "text": "Candidate Image 9:"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_images[8]}"},
                                },
                                {"type": "text", "text": target_match_description},
                            ],
                        }
                    ],
                )
                return completion.choices[0].message.content
            except:
                print(f"Attempt {attempt + 1}/{max_retries} failed. Retrying in {delay} seconds...")
                time.sleep(delay)
        raise Exception("Max retries exceeded. Unable to get a valid response from the model.")
            
        
    def get_result(self, response):
        """
        respond in JSON format as described below:
        [
            { "image_id": "1", "is_same_object": boolean },
            { "image_id": "2", "is_same_object": boolean },
            { "image_id": "3", "is_same_object": boolean },
            { "image_id": "4", "is_same_object": boolean },
            { "image_id": "5", "is_same_object": boolean },
            { "image_id": "6", "is_same_object": boolean },
            { "image_id": "7", "is_same_object": boolean },
            { "image_id": "8", "is_same_object": boolean },
            { "image_id": "9", "is_same_object": boolean }
        ]
        """
        result = {}
        result["reasoning"] = response
        result["results"] = [False] * 9
        try:
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            lenth = len(data)
        except:
            print("Error: response is not in JSON format: ", response)
            return result
        if lenth < 9:
            print("Error: response does not have enough results, fill with False: ", data)
            data += [{"image_id": str(i+1), "is_same_object": False} for i in range(9 - lenth)]
        elif lenth > 9: 
            print("Error: response has too many results, only keep the first 9: ", data)
            data = data[:9]
        try:
            for i in range(9):
                result["results"][i] = data[i]["is_same_object"]
        except:
            print("Error: response does not have the corresponding items: ", response)
        return result
            
    def predict(self, obs_segments):
        """
        obs_segments: np.array ((region_num + 1) * channels, height, width)
        """
        h, w = obs_segments.shape[-2], obs_segments.shape[-1]
        obs_segments = obs_segments.astype(np.uint8).reshape(self.region_num + 1, self.channels, h, w).transpose(0, 2, 3, 1)
        fragments = obs_segments[:-1]
        fragments = [self.arr_to_img(fragment) for fragment in fragments]
        response = self.chat_with_client(fragments)
        result = self.get_result(response)
        return result
