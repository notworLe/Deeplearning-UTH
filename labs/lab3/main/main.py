from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F


if __name__ == '__main__':
    v_model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    v_model = AutoModelForSequenceClassification.from_pretrained(v_model_name)
    v_tokenizer = AutoTokenizer.from_pretrained(v_model_name)

    v_classifier = pipeline("sentiment-analysis", model=v_model_name, tokenizer=v_tokenizer)
    v_list_result = v_classifier([
        "Just have trying, don't end up like me",
        "I'll see you around"
    ])

    for result in v_list_result:
        print(result)

