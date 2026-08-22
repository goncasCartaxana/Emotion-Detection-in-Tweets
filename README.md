# Sentiment Analysis from Tweets

## Overview
This project implements an end-to-end NLP framework that validates human annotation reliability, trains classical machine learning pipelines, and optimizes Large Language Models (LLMs) to classify and analyze sentiments in a Twitter dataset.

## Objectives
The Project has multiple objectives focused on detecting emotions in a dataset of tweets:
1. Evaluate and annotate a dataset, resulting in an Inter-Annotator Agreement analysis.
2. Train various models for tweet classification using preprocessing, text representation techniques (such as SentenceTransformer and TF-IDF), ML modeling (Multi-Layer Perceptron, Naive Bayes, Support Vector Machines), and evaluation metrics (Precision, Recall, F1-Score, and Accuracy).
3. Classify tweet emotions using Large Language Models (LLMs), comparing different models, hyperparameters, and Prompt Engineering techniques.

## Considerations
Throughout the three metas relevant conclusions were drawn:
1. In Meta 3, we explored various models (Llama3.2:3B, Phi3:3.8B, and Llama3.1:8B), temperatures (0.1, 0.3, and 0.6) and prompting techniques (zero-shot, few-shot, chain-of-thought).
2. We understood that sometimes less text is more, low temperature is superior (in this case), and while the size of the model can help performance (Llama3.1:8B), it's not determinant (Phi3:3.8B).
3. In Meta 2, some adopted models achieved very similar accuracies among themselves: MLP and SVM reached 62%, while Naive Bayes reached 61%. We concluded not all text representation methods work with any model. We found that the following combinations are the best: MLP with transformer, SVM and Naive Bayes with TF-IDF.
4. By comparing Meta 2 and Meta 3, it's possible to infer that models trained with the dataset (Meta 2) had better performance than pre-trained general-purpose LLMs (Meta 3). As expected models from Meta 2 were superior, because they were trained to do this task, whereas models from Meta 3, were trained for generic tasks.
5. The project allowed us to apply and consolidate theoretical and practical knowledge in sentiment analysis and language modeling, exploring advanced tools and approaches.
