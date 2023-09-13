import _db
import text_analysisV1 as ta


# 載入資料
def load_data(question):
    stack_items = []
    posts = _db.TEST_DATA.find_one({"question": question})
    for p in posts['posts']:
        data = _db.TEST_OUTER_DATA.find_one({"question.id": p})
        stack_items.append(data)

    return stack_items


if __name__ == "__main__":

    # load data
    i = _db.TEST_DATA.find_one({"category": "CoreLanguage"})
    SO_data = load_data(i['question'])
    # ta.block_ranking(SO_data, [])

    # extract keywords from user question
    analyzer = ta.TextAnalyze()
    qkey, doc = analyzer.content_pre_process(i['question'])

    rank = ta.block_ranking(stack_items=SO_data, question=i['question'])
    print(rank)
    # print(i['question'])
    # print(SO_data)