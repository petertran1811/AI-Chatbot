from openai import OpenAI
import time
client = OpenAI(base_url="https://d74aa5a93536.ngrok-free.app", api_key="minichatbot")
# client = OpenAI(base_url="https://mini-chatbot-core.apero.vn/", api_key="sk-e3sfMIbGQyUabnuy8YdVgCoaisGUhcayA9UR2krPfKSHesl7s28HE6QWH4SrI")
thread = client.beta.threads.create()
print(f"Thread ID: {thread.id}")
from typing_extensions import override
from openai import AssistantEventHandler
 
class EventHandler(AssistantEventHandler):
  def __init__(self, request_start_time: float):
    """
    Khởi tạo EventHandler với thời gian bắt đầu của yêu cầu.
    """
    super().__init__()
    self.request_start_time = request_start_time
    self._first_token_received = False # Cờ để kiểm tra xem đã nhận được token đầu tiên chưa

  @override
  def on_text_created(self, text) -> None:
    # Khi một khối văn bản mới được tạo, đặt lại cờ cho trường hợp nếu có nhiều khối văn bản
    # Tuy nhiên, TTFT chỉ quan tâm đến token đầu tiên của toàn bộ phản hồi.
    # Nên cờ này chủ yếu để đảm bảo chỉ đo TTFT một lần duy nhất cho mỗi phản hồi.
    self._first_token_received = False 
      
  @override
  def on_text_delta(self, delta, snapshot):
    if not self._first_token_received:
      # Nếu đây là delta văn bản đầu tiên được nhận cho lần chạy này
      ttft = time.time() - self.request_start_time
      print(f"\nTTFT: {ttft:.2f} seconds", end="", flush=True)
      print(f"\nassistant > ", end="", flush=True) # In "assistant >" sau khi in TTFT
      self._first_token_received = True
    # client.beta.threads.runs.cancel(thread_id=thread.id, run_id="dsadsadsadsa")
    print(delta.value, end="", flush=True)

first = False
index = 5 
while True:
  msg = input("You: ")
  if first == True:
    message = client.beta.threads.messages.create(
      thread_id=thread.id,
      # thread_id="71393ba5-24a4-4b5f-9b4a-1a54890c60c1",
      role="user",
      content= [
        {
          "type": "text",
          "text": msg
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"https://static.apero.vn/cms-upload/wepu/id_{index}.jpg"
          }
        }
      ]
    )
    # if index == 7:
    #   first = False
    # if index == 5:
    #   index += 2
    # else:
    #   index += 1
    first = False
    
  else:
    message = client.beta.threads.messages.create(
      thread_id=thread.id,
      # thread_id="71393ba5-24a4-4b5f-9b4a-1a54890c60c1",
      role="user",
      content= msg
    )
  
  # Ghi lại thời gian ngay trước khi gửi yêu cầu chạy Assistant
  request_start_time = time.time() 
  
  with client.beta.threads.runs.stream(
    thread_id=thread.id,
    # thread_id="71393ba5-24a4-4b5f-9b4a-1a54890c60c1",
    assistant_id="general_bot",
    event_handler=EventHandler(request_start_time), # Truyền thời gian bắt đầu vào EventHandler
    # metadata={"suggestions": "true"}
  ) as stream:
    stream.until_done()
  
  # Tính toán tổng thời gian phản hồi
  end_time = time.time()
  print(f"\nResponse time: {end_time - request_start_time:.2f} seconds")
  print()

# import os
# import time
# from openai import OpenAI

# client = OpenAI(
#     api_key="0228ae9f-16b7-482a-b017-f550904ad52b",
#     base_url="https://ark.ap-southeast.bytepluses.com/api/v3",
# )

# # Streaming:
# print("----- streaming request -----")

# start_time = time.time()  # thời gian bắt đầu request
# ttft = None  # thời gian đến token đầu tiên
# token_count = 0

# stream = client.chat.completions.create(
#     model="deepseek-v3-250324",
#     messages=[
#         {"role": "system", "content": "You are an AI assistant"},
#         {"role": "user", "content": "What are the common cruciferous plants?"},
#     ],
#     stream=True,
# )

# for chunk in stream:
#     if not chunk.choices:
#         continue
#     delta = chunk.choices[0].delta.content
#     if delta:
#         token_count += 1
#         if ttft is None:
#             ttft = time.time() - start_time  # đo TTFT
#         print(delta, end="")

# end_time = time.time()  # thời gian kết thúc toàn bộ response

# print("\n")
# print(f"TTFT: {ttft:.3f} seconds")
# print(f"Total response time: {end_time - start_time:.3f} seconds")
# print(f"Tokens received: {token_count}")
