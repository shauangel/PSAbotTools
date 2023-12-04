import requests
import config

def embeds(tokens):
    resp = requests.post(config.EM)