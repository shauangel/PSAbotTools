# Stack Overflow Parser
import requests
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath
from bs4 import BeautifulSoup
import json
from datetime import datetime


class StackData:

    # Constants & URL
    STACK_EXCHANGE_API = "https://api.stackexchange.com/2.3"
    MAX_PAGE = 25

    def __init__(self, urls=[]):
        # collect ids
        self.ids = [PurePosixPath(urlparse(unquote(p)).path).parts[2] for p in urls]
        # dict of id & link
        self.links = {k: v for (k, v) in zip(self.ids, urls)}
        # store parsed data
        self.results = []
        self.questions = []
        self.get_questions()
        self.get_answers()

    # Construct request API URL
    def get_api_url(self, page, answers):
        ids = ";".join(self.ids)
        params = "&site=stackoverflow&filter=withbody"
        if answers:
            return self.STACK_EXCHANGE_API + "/questions/" + ids + "/answers?page=" + page + params
        return self.STACK_EXCHANGE_API + "/questions/" + ids + "?page=" + page + params

    # Send requests to Stack Exchange API
    def get_stackexchange_response(self, is_answers):
        # is_answer: True -> for answer posts, False -> for question post
        # Step 1: set page, parameters
        page = 1
        api = self.get_api_url(str(page), is_answers)

        # Step 2: Send requests & Check if needed switch page
        # Parse 1st page
        data = requests.get(api).json()
        data_requests = [data]      # Collects all responses
        # print(data)
        # if it has more pages, construct a new requests for next page
        while data["has_more"]:
            # print("-"*20)
            page += 1
            api = self.get_api_url(str(page), is_answers)
            data = requests.get(api).json()
            data_requests.append(data)
            # print(page)
            # print(data)
            # print("-" * 20)
        return data_requests

    # method: Get full question posts
    def get_questions(self):
        # Step 1: Collect Responses from API
        q_requests = self.get_stackexchange_response(False)

        # Step 2: Clean html tags & Construct StackData
        for r in q_requests:
            try:
                for q in r['items']:
                    # print(q)
                    self.results.append({
                        "link": q["link"],
                        "keywords": [],
                        "tags": q["tags"],
                        "question": {
                            "id": q["question_id"],
                            "title": q["title"],
                            "content": self.get_pure_text(q["body"]),
                            "abstract": ""},
                        "answers": []
                    })
                    self.questions.append(q["title"])
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                print("uh oh, q err")

    # method: Get all questions' full answers
    def get_answers(self):
        # Step 1: Collect all responses
        ans_requests = self.get_stackexchange_response(True)

        # Step 2:
        for r in ans_requests:
            try:
                for ans in r['items']:
                    idx = next((i for (i, d) in enumerate(self.results)
                                if d["question"]["id"] == ans["question_id"]), None)
                    self.results[idx]["answers"].append({
                        "id": ans["answer_id"],
                        "score": ans["score"],
                        "vote": 0,
                        "content": self.get_pure_text(ans["body"]),
                        "abstract": ""
                    })
                    # print(ans)
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                print("uh oh, ")

    @staticmethod
    def get_pure_text(html):
        # get sentences without html tag & code
        soup = BeautifulSoup(html, 'html.parser')
        abstract = [i.text for i in soup.findAll('p')]
        result = " ".join(abstract)
        return result

    # display result
    def get_results(self):
        print(self.results)
        return self.results

    # Save to json file
    def save_results(self, file=""):
        timestamp = datetime.today().isoformat(sep="T", timespec="seconds").replace(":","-")
        with open(file+"StackData" + str(timestamp) + ".json", "w", encoding="utf-8") as f:
            json.dump(self.results, f)


if __name__ == "__main__":
    test_urls = ["https://stackoverflow.com/questions/28461001/python-flask-cors-issue",
                 "https://stackoverflow.com/questions/74583218/python-flask-cors-error-set-according-to-documentation",
                 "https://stackoverflow.com/questions/71950802/flask-cors-work-only-for-first-request-whats-the-bug-in-my-code",
                 "https://stackoverflow.com/questions/71566073/flask-restful-and-angular-cors-error-on-post-method",
                 "https://stackoverflow.com/questions/26980713/solve-cross-origin-resource-sharing-with-flask"]
    parser = StackData(test_urls)
    results = parser.get_results()
    parser.save_results()


