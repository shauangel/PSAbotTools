from bs4 import BeautifulSoup
import requests

class RedditData:

    # Constants & URL
    STACK_EXCHANGE_API = "https://api.stackexchange.com/2.3"
    MAX_PAGE = 25

    def __init__(self):
        self.i = 0

if __name__ == "__main__":
    test_url = ["https://www.reddit.com/r/reactjs/comments/l192dn/how_to_deal_with_cors_error_in_react_and_flask_app/",
                "https://www.reddit.com/r/flask/comments/15lzkxx/i_have_tunneled_my_flask_app_to_ngrok_that_now_is/",
                "https://www.reddit.com/r/flask/comments/knhg40/cors_problem_react_flask/",
                "https://www.reddit.com/r/learnpython/comments/jvsfsq/getting_cors_errors_when_running_flask_app_from/",
                "https://www.reddit.com/r/flask/comments/ttoyqp/flask_cors_policy_need_help/"]
    resp = requests.get(test_url[0])
    # print(resp.json)
    parser = BeautifulSoup(resp.text, 'html.parser')
    text = parser.find('body').get_text()
    print(text)


