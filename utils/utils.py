import re
import math
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import torch

import wikipediaapi
wiki_wiki = wikipediaapi.Wikipedia('en')

def clean_string(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'`.]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip()

def get_text_from_wiki(text, n_sents=2):
    text = text.replace('-', ' ')
    page_py = wiki_wiki.page(text)
    paragraph = sent_tokenize(page_py.summary)
    if len(paragraph) == 0:
        return text
    elif len(paragraph) <= n_sents:
        return " ".join(paragraph)
    else:
        return " ".join(paragraph[:n_sents])
    
def normalizeAdjacency(W):
    assert W.size(0) == W.size(1)
    d = torch.sum(W, dim = 1)
    d = 1/torch.sqrt(d)
    D = torch.diag(d)
    return D @ W @ D 

def get_embedding_from_wiki(sbert, text, n_sent=1):
    text = get_text_from_wiki(text, n_sent)
    embedding = sbert.encode(text, convert_to_tensor=True)
    return embedding