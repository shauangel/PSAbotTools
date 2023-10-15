# basic flask framework
from flask import Flask
from views import register_blueprint

# CORS validation
from flask_cors import CORS

# set Flask app
app = Flask(__name__)
CORS(app)
register_blueprint(app)
