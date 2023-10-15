from flask import request, Blueprint, jsonify
from models import text_analysisV0
from models import text_analysisV1
from models import text_analysisV2

# register api
text_analysis_api = Blueprint("text_analysis_api", __name__)


@text_analysis_api.route("/data_clean", methods=['POST'])
# content pre-process for single doc
def data_clean():
    data = request.get_json()
    try:
        print("Cleaning Document...")
        try:
            if data['version'] == 0:
                ta = text_analysisV0.TextAnalyze()
            elif data['version'] == 1:
                ta = text_analysisV1.TextAnalyze()
            else:
                ta = text_analysisV2.TextAnalyze()
        except KeyError:
            ta = text_analysisV2.TextAnalyze()
        tokens, doc = ta.content_pre_process(data['content'])
        response = {"token": tokens}
    except Exception as e:
        response = {"error": e.__class__.__name__ + " : " + e.args[0]}
    return jsonify(response)


@text_analysis_api.route("/block_analysis", methods=['POST'])
# block analysis
def block_analysis():
    data = request.get_json()
    try:
        print("Processing Block Analysis ...")
        try:
            print("Using Version " + str(data['version']) + " Method ...")
            if data['version'] == 0:
                ranks = text_analysisV0.block_ranking(stack_items=data['items'], qkey=data['qkey'])
            elif data['version'] == 1:
                ranks = text_analysisV1.block_ranking(stack_items=data['items'], qkey=data['qkey'])
            else:
                ranks = text_analysisV2.block_ranking(stack_items=data['items'], qkey=data['qkey'])
        except KeyError:
            print("Using default version 2 ...")
            ranks = text_analysisV2.block_ranking(stack_items=data['items'], qkey=data['qkey'])
        response = {"ranks": ranks}
    except Exception as e:
        response = {"error": e.__class__.__name__ + " : " + e.args[0]}
    return jsonify(response)

