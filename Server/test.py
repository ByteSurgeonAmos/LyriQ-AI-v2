import os
test_path = "C:/Users/amo$/rap-bot/lyriq_ai_server/Server/data/processed_index/test.txt"
os.makedirs(os.path.dirname(test_path), exist_ok=True)
with open(test_path, "w") as f:
    f.write("Permission test")
