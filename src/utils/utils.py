from langchain_core.messages import HumanMessage, RemoveMessage, AIMessage, SystemMessage

def convert_message_to_langgraph_message(message: dict) -> HumanMessage | AIMessage:
    """
    Convert a single message to a LangGraph message.
    """
    if message["role"] == "user":
        # content = []
        # content.append(
        #     {"type": "text", "text": message["content"][0]["text"]["value"]}
        # )
        # if len(message["content"]) > 1 and message["content"][1]["type"] == "image_url":
        #     content.append(
        #         {"type": "image_url", "image_url": {"url": message["content"][1]["image_url"]["url"]}}
        #     )
        content = []
        for content_item in message["content"]:
            if content_item["type"] == "text":
                content.append(
                    {"type": "text", "text": content_item["text"]["value"]}
                )
            elif content_item["type"] == "image_url":
                content.append({"type": "image_url", "image_url": {"url": content_item["image_url"]["url"]}})
        return HumanMessage(content=content)
    elif message["role"] == "assistant":
        return AIMessage(content=message["content"][0]["text"]["value"])
    elif message["role"] == "system":
        return SystemMessage(content=message["content"][0]["text"]["value"])
    else:
        raise ValueError(f"Unknown role: {message['role']}")


import os
import json
import csv
import pandas as pd
from PyPDF2 import PdfReader

def open_any_file_and_save(filepath, output_txt='output.txt'):
    """
    Open a wide range of common file types (excluding images and zip),
    then save its content to a text file.
    
    filetype:
      - 'text'
      - 'json'
      - 'csv'
      - 'excel'
      - 'pdf'
      - 'binary'
      - 'error'
    
    Returns (filetype, data)
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    try:
        if ext in ['.txt', '.md', '.log']:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        
        elif ext == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                obj = json.load(f)
                content = json.dumps(obj, indent=2, ensure_ascii=False)
        
        elif ext == '.csv':
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                content = json.dumps(rows, indent=2, ensure_ascii=False)
        
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
            content = df.to_csv(index=False)
        
        elif ext == '.pdf':
            with open(filepath, 'rb') as f:
                reader = PdfReader(f)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() or ''
                content = text
        
        else:
            # fallback: binary mode (just show size)
            with open(filepath, 'rb') as f:
                data = f.read()
                content = f"[Binary file with {len(data)} bytes]"
        
        # save to output_txt
        with open(output_txt, 'w', encoding='utf-8') as out:
            out.write(content)
        
        return ext, content
    
    except Exception as e:
        return 'error', str(e)

from json import JSONDecodeError
from langchain_core.utils.json import parse_json_markdown
def parse_result(text: str, partial: bool) -> dict | str:
    text = text.strip()
    if partial:
        try:
            return parse_json_markdown(text)
        except JSONDecodeError:
            return ""
    else:
        try:
            return parse_json_markdown(text)
        except JSONDecodeError as e:
            msg = f"Invalid json output: {text}"
            return ""