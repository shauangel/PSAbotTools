import requests
import time

# VM Host: 192.168.100.40:8080
# Host name: psabot-toolbox.soselab.tw

if __name__ == "__main__":
    local_test_url = "http://127.0.0.1:8000"
    test_sen = "Hi, just testing my APIs."
    json = {"keywords": ["python", "flask", "framework"],
            "result_num": 10,
            "page_num": 0}
    # resp = requests.post(url="http://192.168.100.40:8080/api/search",
    #                     json=json)
    #　print(resp.json())

    q = "How to use flask"
    # Step 1. Search via Custom JSON Search API
    result_page = requests.post(url=local_test_url + "/api/search",
                                json={"keywords": ["python", "use", "flask"],
                                      "result_num": 10,
                                      "page_num": 0})
    while result_page.status_code == 204:
        time.sleep(10)
        if result_page.status_code == 200:
            break

    # Step 2. Fetch data
    if result_page.status_code == 200:
        print("hi")
        stack_items = requests.post(url=local_test_url + "/SO_api/get_items",
                                    json={"urls": result_page.json()['result']})
    while stack_items.status_code == 204:
        time.sleep(10)
        if stack_items.status_code == 200:
            break
    print(stack_items.json())
    # Step 3. block analysis
    resp = requests.post(url=local_test_url + "/api/block_analysis",
                         json={"items": stack_items['items'], "q": q}).json()
    print(resp)
