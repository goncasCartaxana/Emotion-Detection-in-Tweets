
# Meta 2: Classification Models
Train machine learning models for emotion classification based on the TwitterDialoguePT dataset (Carvalho et al., 2022) and its sentiment scores annotation. 
The dataset consists of dialogues in Portuguese, with labeled data to train a supervised learning model.

To achieve this goal, a pipeline was created for importing and filtering data, input representations for various machine learning models, and metrics. 
Throughout the report, the various choices, such as pre-processing options, models, and analysis of metrics, are explained.

## Pre-Processing
There are a total of 4190 lines (i.e., tweets) in the entire dataset. 
Specifically, there are 2285 lines in the training set and 1905 lines in the test set.

### Removal of Dialogs with NaN Values
- In the training set, there are no NaN values. In the test set, there were Nan values.
- All tweets associated with a `dialog_ID` that contains at least one tweet with a NaN value were removed. 
- This issue arose due to some tweets not being classified by one of the annotators.

### Use of Regular Expressions (Regex)
We used Regex to replace terms in tweets to increase model generalization. Our goal is to ensure that the model focuses on relevant content. To achieve this objective, we:
- Substitution of Regex Entities: To avoid classifying based on entity identification (e.g., @RTP or @joana), entities are replaced with a neutral term "@user," finding all tweets starting with "@."
- Substitution of Regex Links: To eliminate noise, any links, i.e., excerpts starting with "http" or "www," were removed.

### Training, Testing, and Data Splitting
We decided to mix the training and test data for two reasons:
1. Increase the size of the training data, which will result in a better model due to the common practice being around 20% for testing data.
2. Mix annotated data by students and provided data by the professor with the intention of not training with one dataset and evaluating with another because annotations may have subjectivity and different circumstances.

### Text Representation
For representing text, we chose the following methods:
- TF-IDF (Term frequency-Inverse document frequency)
- SentenceTransformer

TF-IDF
- We considered TF-IDF superior to TF and bag-of-words because it assigns more weight to relevant terms and less to common words.
- TF-IDF is based on a statistical approach and evaluates the importance of words based on their frequency in the document and the corpus.

In the TF-IDF, parameters were adjusted to optimize text representation and improve model efficiency:
- min\_df: Removes rare terms, reducing noise.
- max\_df: Excludes very frequent terms since they are less informative.

SentenceTransformer
- The SentenceTransformer generates dense representations that capture complex semantic relationships and nuances of meaning. 
- A good choice to represent an entire sentence in a latent space. We thought it would perform better than, for example, the average latent space of all words using a transformer.
- Preferred using SentenceTransformer over word2vec or GloVe because we believed it captured semantic nuances better.

## Model for Classification
For supervised learning classification models, we selected:
- Multilayer Perceptron (MLP)
- Naive Bayes
- Support Vector Machines (SVM)

We chose models of different natures to have a richer set of comparisons:
- MLP: A neural network
- Bayesian Network: A probabilistic model
- SVM: A high-dimensional mathematical model.

MLP
- Leverages its multiple hidden layers to capture complex patterns and is particularly beneficial for dense vectors and continuous values. 
- With TF-IDF, implementation was direct, and better results were found with 5 layers. 
- With Transformer Embeddings, 4 layers were used, and the StandardScaler() function was used since scaling embeddings helps the model learn more effectively.

Naive Bayes
- The naive Bayes classifier assumes independence among features, demonstrating good results with TF-IDF as it configures a sparse matrix of values. 
- However, it generally does not work well with dense vectors because the Naive Bayes expects counts or frequencies, and the embeddings contain negative values. 
- Since the Sentence Transformer generates embeddings, we opted for Gaussian Naive (GaussianNB) that can handle continuous values.

SVM
- The goal of Support Vector Machines is to find the optimal hyperplane that separates data in different classes, maximizing the margin between them.
- Like the Naive Bayes, it obtained good results with sparse vectors and thus had good results with TF-IDF. 
- In the case of embeddings, we had to normalize the values using the StandardScaler() function to solve the problems found.

## Evaluation Metrics
Chosen metrics: Precision, Recall, F1-Score, and Accuracy.
The dataset is imbalanced due to the predominance of the real category being "0." 
Here are the metrics for the 3 models with the 2 input methods:

| Accuracy    | NB  | MLP | SVM |
| ----------- | --- | --- | --- |
| Transformer | 53% | 62% | 58% |
| TF-IDF      | 61% | 59% | 62% |


## Discussion
### Pre-Processing
Of all the data processing approaches adopted, the "Training, Testing, and Data Splitting" section had the most significant impact on the final results, leading to an increase of 8% in accuracy in some models during repeated tests throughout the project.

#### Model
The models were able to achieve similar accuracies: MLP and SVM reached 62%, while Naive Bayes reached 61%.

#### Inputs
Text representation (inputs) performed well for the selected models: MLP with transformer, SVM, and Naive Bayes with TF-IDF.

### Evaluation
In a classification problem with 4 categories, a random guess would be correct once every 4 times (an average of 25%). 
A model that classifies 4 categories needs to have more than 25% accuracy for its guess to better than random. 
By obtaining 62% accuracy, we achieved models with significantly good performance.

### Considerations
The results demonstrate that both classical approaches (SVM, Naive Bayes) and modern ones (MLP with transformer) can be effective for this task. 
There are various possible solutions to the problem, and many factors were not considered, such as speed, algorithm complexity, and interpretability. 
We suppose transformers will have better performance. We await meta 3 to test our assumptions.
