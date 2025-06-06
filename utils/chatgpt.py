# utils/chatgpt.py
# (C) Yoshi Sato <satyoshi.com>

from __future__ import annotations
import openai, json, os
from utils.utils import *
from utils.llm_async import ask_async         # ★ NEW ★

with open("utils/chatgpt/openai.json", "r") as f:
    g_openai_config: dict = json.load(f)


class ChatGPT:
    """Wrapper that can run *synchronously* (old __call__) or
       *asynchronously* (new ask_async)."""

    def __init__(self, config: dict, arglist):
        # -- API / model params ------------------------------------------------
        if "OPENAI_API_KEY" in os.environ:
            self.api_key: str = os.environ["OPENAI_API_KEY"]
        else:
            self.api_key: str = config["access_token"]
        openai.api_key = self.api_key

        self.model       = config["model"]
        self.temperature = float(config["temperature"])
        self.top_p       = float(config["top_p"])
        self.max_tokens  = int(config["max_tokens"])

        # ---------------------------------------------------------------------
        self.messages: list = []
        self.num_agents: int = arglist.num_agents

        if self.num_agents == 1:
            with open("utils/prompts/single_agent_instruction.txt") as f:
                instruction = f.read()
            with open("utils/prompts/single_agent_example.txt") as f:
                example = f.read()
        elif self.num_agents == 2:
            with open("utils/prompts/multi_agent_instruction.txt") as f:
                instruction = f.read()
            with open("utils/prompts/multi_agent_example.txt") as f:
                example = f.read()
        else:
            raise ValueError(f"num_agents must be 1 or 2, not {self.num_agents}")

        # ChatML conversation seed
        self.messages += [
            {"role": "system", "content": "You are a Python programmer. Help me write code in Python."},
            {"role": "user",   "content": instruction},
            # one-shot example
            {"role": "system", "name": "example_user",      "content": "Make a lettuce salad."},
            {"role": "system", "name": "example_assistant", "content": example}
        ]

    # --------------------------------------------------------------------- #
    # OLD synchronous call (kept for backward compatibility)
    # --------------------------------------------------------------------- #
    def __call__(self, message: str) -> str:
        self.messages.append({"role": "user", "content": message})
        result = self._execute_sync()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def _execute_sync(self) -> str:
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stream=True)                           # synchronous streamer
        chunks = []
        for chunk in completion:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                text = str(delta["content"])
                print(colors.YELLOW + text + colors.ENDC, end="", flush=True)
                chunks.append(text)
        return "".join(chunks)

    # --------------------------------------------------------------------- #
    # NEW non-blocking helper
    # --------------------------------------------------------------------- #
    def ask_async(self, message: str):
        """
        Fire the request *without* blocking.  Returns a Queue.  Poll it with
        `q.get_nowait()` from your game loop; when data arrives it is either
        `str` (the reply) or `RuntimeError`.
        """
        all_msgs = self.messages + [{"role": "user", "content": message}]
        return ask_async(
            all_msgs,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stream=False)            # disable streaming inside worker
