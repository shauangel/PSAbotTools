import requests

# VM Host: 192.168.100.40:8080
# Host name: psabot-toolbox.soselab.tw

if __name__ == "__main__":
    test_sen = "Hi, just testing my APIs."
    json = {"keywords": ["python", "flask", "framework"],
            "result_num": 10,
            "page_num": 0}
    resp = requests.post(url="http://192.168.100.40:8080/api/search",
                         json=json)
    print(resp.json())
