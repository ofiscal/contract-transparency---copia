# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:51:08 2024

@author: usuario
"""

from transformers import pipeline

# Load pre-trained model
nlp = pipeline("zero-shot-classification",
                      model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")


def is_sustainable(text):
    # Define sustainability-related keywords
    keywords = ["agua","energia","biodiversidad","cambio climático",
                "residuos sólidos","calidad atmosférica","calidad del aire",
                "generación de empleo","costo total de propiedad","otros"]

    # Split the text into sentences
    sentences = text.split('.')

    # Analyze each sentence
    for sentence in sentences:
        # If any of the keywords are in the sentence, analyze it

          result = nlp(sentence,candidate_labels=keywords)
          print(f'Sentence: {sentence},', result)

# Test the function
text = "estamos procurando la reduccion de gases con efecto invernadero"
is_sustainable(text)





