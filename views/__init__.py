from .text_analysis_api import text_analysis_api
from .stack_data_api import stack_data_api
from .data_manager_api import data_manager_api

blueprint_prefix = [
                    (text_analysis_api, "/api"),
                    (stack_data_api, "/api"),
                    (data_manager_api, "/api")
                    ]


def register_blueprint(app):
    for blueprint, prefix in blueprint_prefix:
        app.register_blueprint(blueprint, url_prefix=prefix)
    return app
