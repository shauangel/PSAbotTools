from flask import request, Blueprint, jsonify
from models import outer_search

# register api
outer_search_api = Blueprint("outer_search_api", __name__)


@outer_search_api.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    try:
        result = outer_search.outer_search(keywords=data["keywords"],
                                           result_num=data["result_num"],
                                           page_num=data["page_num"])
        response = {"result": result}
    except Exception as e:
        response = {"error": e.__class__.__name__ + " : " + e.args[0]}
    return jsonify(response)