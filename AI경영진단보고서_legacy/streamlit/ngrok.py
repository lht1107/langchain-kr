import os
from dotenv import load_dotenv
from pyngrok import ngrok

load_dotenv()

ngrok_token = os.getenv("NGROK_TOKEN")

ngrok.set_auth_token(ngrok_token)

# public_url = ngrok.connect('http://localhost:8501')
# print(f"Stremlit URL: {public_url}")

# Windows의 경우 경로 설정 (이 부분을 주석 해제해야 함)
config_dir = os.path.expanduser('~/.ngrok2')
os.makedirs(config_dir, exist_ok=True)

# ngrok 설정 내용
ngrok_config = f'''version: "2"
authtoken: {ngrok_token}
tunnels:
  app1:
    addr: 8501
    proto: http
  app2:
    addr: 8502
    proto: http
'''

# 설정 파일 저장 (경로 수정)
config_path = os.path.join(config_dir, 'ngrok.yml')
with open(config_path, 'w') as config_file:
    config_file.write(ngrok_config)
    print("success")
