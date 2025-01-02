import base64
import io
import json
import requests
from .imagefunc import *


deepseek_model_list = ['deepseek-chat']

class LS_DeepSeek_API:

    def __init__(self):
        self.NODE_NAME = 'DeepSeekAPI'
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "model": (deepseek_model_list,),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192, "step": 1}),
                "temperature": ("FLOAT", {"default": 1, "min": 0, "max": 2, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.1}),
                "presence_penalty": ("FLOAT", {"default": 0, "min": -2, "max": 2, "step": 0.1}),
                "frequency_penalty": ("FLOAT", {"default": 0, "min": -2, "max": 2, "step": 0.1}),
                "history_length": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": False}),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "history": ("DEEPSEEK_HISTORY",),
            }
        }

    RETURN_TYPES = ("STRING", "DEEPSEEK_HISTORY",)
    RETURN_NAMES = ("text", "history",)
    FUNCTION = 'run_deepseek_api'
    CATEGORY = '😺dzNodes/LayerUtility'

    def run_deepseek_api(self, model, max_tokens, temperature, top_p, presence_penalty, frequency_penalty,
                         history_length, system_prompt, user_prompt, history=None):

        api_key = get_api_key('deepseek_api_key')

        ret_text = ""

        if history is not None:
            messages = history["messages"]
            messages = messages[-history_length *2:]
        else:
            messages = [{"role": "system", "content": system_prompt}]
        task = {"role": "user", "content": user_prompt}
        messages.append(task)


        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            temperature=temperature,
            top_p=top_p,
            stream=False
        )

        ret_message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": ret_message})


        log(f"{self.NODE_NAME}: {model} response is:\n\033[1;36m{ret_message}\033[m")
        return (ret_message, {"messages": messages},)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: DeepSeekAPI": LS_DeepSeek_API,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: DeepSeekAPI": "LayerUtility: DeepSeek API",
}