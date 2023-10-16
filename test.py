import requests


if __name__ == "__main__":
    test_sen = "Hi, just testing my APIs."
    resp = requests.post(url="http://127.0.0.1:8000/SO_api/get_items",
                         json={"urls": ["https://stackoverflow.com/questions/28461001/python-flask-cors-issue"] , "result_num":5, "page_num":0})
    print(resp.json())
