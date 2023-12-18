import requests
from models import config
import numpy


# Request embeddings from embedding server
def embeds(tokens):
    resp = requests.post(config.EMBEDDINGS_URL,
                         json={"doc_list": tokens})
    eb = resp.json()
    return numpy.array(eb)
