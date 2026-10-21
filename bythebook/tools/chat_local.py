"""Chat with the locally served vision-language model from the terminal.

    python -m bythebook.tools.chat_local                 # text chat
    python -m bythebook.tools.chat_local photo.jpg ...   # attach one or more images to the first message

Type a message and press ENTER; an empty line quits. The server is the one started by
bythebook/tools/serve_local_vlm.sh (http://127.0.0.1:8000/v1).
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

from openai import OpenAI


def main():
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
    model = client.models.list().data[0].id
    images = [Path(p) for p in sys.argv[1:]]
    history = []
    print(f"chatting with {model}; empty line to quit" + (f"; {len(images)} image(s) attached" if images else ""))
    while True:
        try:
            text = input("\nyou: ").strip()
        except EOFError:
            break
        if not text:
            break
        content = [{"type": "text", "text": text}]
        for p in images:
            data = base64.standard_b64encode(p.read_bytes()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
        images = []                                  # only with the first message
        history.append({"role": "user", "content": content})
        print("model: ", end="", flush=True)
        reply = ""
        with client.chat.completions.stream(model=model, messages=history, max_tokens=2000, temperature=0.7) as stream:
            for event in stream:
                if event.type == "content.delta":
                    print(event.delta, end="", flush=True)
                    reply += event.delta
        print()
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
