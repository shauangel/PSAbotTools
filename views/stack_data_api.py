from flask import request, Blueprint, jsonify
from models import stack_overflow_parser

# register api
stack_data_api = Blueprint("stack_data_api", __name__)


@stack_data_api.route("/get_items", methods=['POST'])
def get_items():
    data = request.get_json()
    # ip = request.remote_user
    # print("Test ip detecting: " )
    # print(ip)
    try:
        stack_data = stack_overflow_parser.StackData(data["urls"])
        response = {"items": stack_data.get_results()}
    except Exception as e:
        response = {"error": e.__class__.__name__ + " : " + e.args[0]}
    return jsonify(response)


@stack_data_api.route("/download", methods=["POST"])
# download detail of SO posts to local file system
def download():
    data = request.get_json()
    # ... still working ...
    return jsonify([])



# future work: flask-cache -> temporary saving user's request & fetching results
