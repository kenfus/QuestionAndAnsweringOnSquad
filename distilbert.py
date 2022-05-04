from transformers import Trainer, default_data_collator, AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, pipeline
from datasets import load_dataset, load_metric
import torch
import numpy as np
from tqdm.auto import tqdm
import collections

class DistilBERT():

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("./test-squad-trained_distilbert")
        self.model = AutoModelForQuestionAnswering.from_pretrained("./test-squad-trained_distilbert")

    def predict(self, question: str, context: str):
        """
        Returns a dictionary containing score, start index , end index and answer
        """
        predictor = pipeline('question-answering', model=self.model, tokenizer=self.tokenizer)
        return predictor({'question': question, 'context': context})


if __name__ == '__main__':
    testmodel = DistilBERT()
    print(testmodel.predict("who am I?", "My name is Roman"))