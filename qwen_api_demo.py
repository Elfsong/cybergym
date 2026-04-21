import os
from pathlib import Path

from openai import OpenAI


def _load_dotenv() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


_load_dotenv()

client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def call_with_openai_client(messages):
    reasoning_content = "" 
    answer_content = ""     
    is_answering = False   
    is_thinking = False

    completion = client.chat.completions.create(
        model="qwen3-32b",
        messages=messages,
        stream=True,
        extra_body = {
            # enable thinking, set to False to disable
            "enable_thinking": True,
            # use thinking_budget to contorl num of tokens used for thinking
            # "thinking_budget": 4096
        }
        # stream_options={
        #     "include_usage": True
        # }
    )

    for chunk in completion:
        if not chunk.choices:
            print("\nUsage:")
            print(chunk.usage)
        else:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                if not is_thinking:
                    print("\n" + "=" * 20 + "Thinking" + "=" * 20)
                    is_thinking = True
                print(delta.reasoning_content, end='', flush=True)
            else:
                if delta.content and is_answering is False:
                    print("\n" + "=" * 20 + "Answer" + "=" * 20 + "\n")
                    is_answering = True
                if is_answering:
                    print(delta.content, end='', flush=True)
               
        
if __name__ == '__main__':
    messages = [
        {
            "role": "user",
            "content": "who are you？"
        }
    ]
    call_with_openai_client(messages)