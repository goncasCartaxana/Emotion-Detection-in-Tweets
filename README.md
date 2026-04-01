# Emotion Detection in Tweets (EDT)

## Introduction
The Project has multiple objectives focused on detecting emotions in a dataset of tweets:

### Objectives
1. Evaluate and annotate a dataset, resulting in an Inter-Annotator Agreement analysis.
2. Train various models for tweet classification using preprocessing, text representation techniques (such as SentenceTransformer and TF-IDF), ML modeling (Multi-Layer Perceptron, Naive Bayes, Support Vector Machines), and evaluation metrics (Precision, Recall, F1-Score, and Accuracy).
3. Classify tweet emotions using Large Language Models (LLMs), comparing different models, hyperparameters, and Prompt Engineering techniques.

## Meta 1: Annotations and Data Analysis
A sample of Tweets was manually annotated with sentiment scores ranging from [-2, 1]. The dataset used for this project is a portion of the TwitterDialogueSAPT dataset (Carvalho et al., 2022), which consists of dialogues in Portuguese gathered from Twitter. The data includes interactions related to e-commerce services, television, and healthcare.

Our specific subset involved dialogues related to Netflix (@NetflixPT), RTP Notícias (@RTPNoticias), and Vodafone (@VodafonePT). The corpus contains informal conversations with abbreviations, internet slang, and English expressions.

### Annotations
The annotation process was individual. Emerging criteria for annotators included:
- All humorous phrases tend to be positive.
- A distinction between negative (-2 and -1) categories is related to the perceived level of user dissatisfaction or aggressive tone of vocabulary used.
- Minimal sadness or enthusiasm is considered positive;
- The neutral (0) value usually indicates that the message is merely informative.

### Data Analysis
1. Data cleaning removed NaN values in `dialog_ID`.
2. Two popular NLP libraries were used for corpus analysis (NLTK and spaCy).
3. Stopwords and punctuation were eliminated, and the total number of tokens and of unique tokens were determined using both tools.
4. NLTK identified common tokens, while spaCy offered functions for entity and link identification.

### Analysis of Annotations
#### Analysis of categories by annotator
The values of concordance indicate a substantial level of agreement. These values are within the range of substantial consistency between annotators, according to the Kappas. According to the Krippendorff Alpha, the level of agreement is low.

#### Analysis of categories for responses to entities
The Vodafone has more responses with extremely negative (-2) sentiments, RTP has slightly negative (-1) sentiments, and Netflix has mostly neutral sentiments. The majority of the data consists of responses related to Netflix, followed by fewer responses for Vodafone and RTP. Negative sentiment associated with Vodafone is justified due to user dissatisfaction with the service provided. Similarly, negative sentiment towards RTP can be attributed to content about war.

### Discussion
The difference in sentiments among entities: Vodafone received a high volume of extremely negative responses, whereas Netflix was generally categorized as neutral. RTP received slightly negative responses. The reasons include the content of tweets from the entities and dissatisfaction with the services provided (particularly from Vodafone).

### Limitations
The limited size of the dataset (100 Tweets) may restrict the generalization of results for a broader analysis of sentiments in digital interactions.

## Meta 2: Classification Models
Train machine learning models for emotion classification based on the TwitterDialoguePT dataset (Carvalho et al., 2022) and its sentiment scores annotation. The dataset consists of dialogues in Portuguese, with supervised training data for model supervised learning.

To achieve this goal, a pipeline was created for importing and filtering data, input representations for various machine learning models, and metrics. Throughout the report, we explain the various choices, such as pre-processing options, models, and analysis of metrics.

### Pre-Processing
There are a total of 4190 lines (i.e., tweets) in the entire dataset. 
Specifically, there are 2285 lines in the training set and 1905 lines in the test set.

#### Removal of Dialogs with NaN Values
- In the training set, there are no NaN values. In the test set, there were Nan values.
- All tweets associated with a `dialog_ID` that contains at least one tweet with a NaN value were removed. 
- This issue arose due to some tweets not being classified by one of the annotators.

#### Use of Regular Expressions (Regex)
We used Regex to replace terms in tweets to increase model generalization. Our goal is to ensure that the model focuses on relevant content. To achieve this objective, we:
- Substitution of Regex Entities: To avoid classifying based on entity identification (e.g., @RTP or @joana), entities are replaced with a neutral term "@user," finding all tweets starting with "@."
- Substitution of Regex Links: To eliminate noise, any links, i.e., excerpts starting with "http" or "www," were removed.

#### Training, Testing, and Data Splitting
We decided to mix the training and test data for two reasons:
1. Increase the size of the training data, which will result in a better model due to the common practice being around 20% for testing data.
2. Mix annotated data by students and provided data by the professor with the intention of not training with one dataset and evaluating with another because annotations may have subjectivity and different circumstances.

#### Text Representation
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

### Model for Classification
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

### Evaluation Metrics
Chosen metrics: Precision, Recall, F1-Score, and Accuracy.
The dataset is imbalanced due to the predominance of the real category being "0." 
Here are the metrics for the 3 models with the 2 input methods:

| Accuracy    | NB  | MLP | SVM |
| ----------- | --- | --- | --- |
| Transformer | 53% | 62% | 58% |
| TF-IDF      | 61% | 59% | 62% |


### Discussion
#### Pre-Processing
Of all the data processing approaches adopted, the "Training, Testing, and Data Splitting" section had the most significant impact on the final results, leading to an increase of 8% in accuracy in some models during repeated tests throughout the project.

#### Model
The models were able to achieve similar accuracies: MLP and SVM reached 62%, while Naive Bayes reached 61%.

#### Inputs
Text representation (inputs) performed well for the selected models: MLP with transformer, SVM, and Naive Bayes with TF-IDF.

#### Evaluation
In a classification problem with 4 categories, a random guess would be correct once every 4 times (an average of 25%). A model that classifies 4 categories needs to have more than 25% accuracy. By obtaining 62% accuracy, we achieved models with significantly good performance.

#### Considerations
The results demonstrate that both classical approaches (SVM, Naive Bayes) and modern ones (MLP with transformer) can be effective for this task. 
There are various possible solutions to the problem, and many factors were not considered, such as speed, algorithm complexity, and interpretability. 
We suppose transformers will have better performance. We await meta 3 to test our assumptions.

## Meta 3: Prompt Engineering with LLMs
In this final meta, we explore Large Language Models (LLMs), focusing on Prompt Engineering techniques. 
The goal is to understand how LLMs can be used to classify sentiment by crafting appropriate prompts that guide their reasoning process.

### Overview of LLMs
Large Language Models are powerful AI models that can process a vast amount of textual data, with the processing time and computational complexity increasing non-linearly (usually $O(n^2)$). For instance, in this project, we observed that a system prompt containing too many tokens significantly increased the processing time.

### Prompting
Prompt Engineering involves designing prompts to guide LLMs in performing specific tasks. Here are some benefits of using chain-of-thought (CoT) approach:
* Improved performance (accuracy)
* Enhanced interpretability (more transparent reasoning process)
* Better generalization (helps with complex tasks)

We will focus on the following factors:
1. LLM sizes (up to 8B):
	* Models
	* Temperature
2. System Prompt:
	* Description: Defining the problem (Role)
	* Context: Additional information that aids in the task (Criteria)
	* Format: How we ask for the response (Chain-of-Thought + Regex)
3. Capturing sentiment value using Regex
4. Iterating up to three times when Regex fails to identify a valid value

### Conjuncts of Experiences

#### Experiment 1 - Baseline
We started by testing simple prompts with different structures, aiming to explain the LLM's objective: classify the sentiment of tweets. The best result was obtained using the baseline 1 prompt; therefore, we decided to use it for the subsequent experiments. This was using a temperature of 0.1.

#### Experiment 2 - Temperature
We tested three new temperature values (0.2, 0.3, and 0.6) regarding temperature. A temperature of 0.3 was closest to the previous obtained accuracy. However, the temperature of 0.1 still performed better.

#### Experiment 3 - Criteria
With CoT and context, we found that shorter and more objective phrases resulted in improved performance, leading to an increase in accuracy by 11.25%. This improvement marked the best result achieved so far with a 50% accuracy rate. The devil advocate approach failed to yield any significant improvement, as the LLM's creativity did not positively impact the performance.

#### Experiment 4 - Models (LLMs Larger)
We concluded that the quality of the system prompt is most crucial in determining the model's response quality. However, other factors such as the amount of text in the prompt affect the efficiency of processing. Hence, we decided to use the baseline 1 prompt for testing larger models since it presented a slight reduction in accuracy compared to COT + context (-text).

#### Discussion
We considered not using tweets grouped by DialogID but providing better context to the LLM could help classify sentiments more accurately. Additionally, we did not explore other parameters of LLMs like Top K, Top P, and Min P, which determine which tokens are sampled during processing. Comparing the base model with a fine-tuned one, as well as different quantizations of models, is another aspect worth exploring.

We also noticed tweets where the models refused to classify. Remedies for this issue could be:
1. Preprocessing to replace insults with generic terms (e.g., insult) or
2. Using an uncensored model.

Examples:
* Tweet(censored): "Dados moveis da @user com defeito, p@t@ m3rd@"
* Answered with: "Desculpe, mas não posso cumprir esse pedido."

Strategies with better results:
* Surprisingly, the zero-shot strategy worked well compared to the strategies using chain-of-thought.
* A temperature of 0.1 performed better than expected even with chain-of-thought.

Performance depends on the quality of the data
- We found examples where tweets seemed misclassified. For instance, the following tweets should have been classified as 1:
- "@user Muito obrigada, @user. ❤ " is classified as -2.
- "@user finalmente ♥️ " is classified as 0.
- There are tweets identical with different classifications. Example: "Bolo rei é péssimo! 😠 " is classified both as -1 and -2.

### Considerations
In this final meta, we explored various aspects of LLMs and Prompt Engineering to classify sentiment in tweets. Through our experiments, we gained valuable insights and made some general conclusions, reflections, and recommendations for future research:

#### General Conclusions
1. **Quality of the system prompt**: Our results demonstrated that the quality of the system prompt significantly affects the LLM's performance. A well-crafted prompt helps guide the model towards the desired task and improves its accuracy.
2. **Temperature settings**: We discovered that a lower temperature value generally led to better performance, although there were exceptions where higher temperatures worked better. Future research could explore more fine-tuning of temperature values for specific tasks.
3. **The role of context**: Although we did not explicitly use DialogID during our experiments, providing better context to the LLM could potentially help improve its ability to classify sentiments accurately.
4. **Exploring other parameters**: We only explored a few parameters such as temperature and prompt format; future research should delve into other parameters like Top K, Top P, and Min P that determine which tokens are sampled during processing.
5. **Preprocessing and censorship**: When dealing with tweets containing offensive language or insults, pre-processing to replace these terms with generic ones or using an uncensored model could help address the issue of LLMs refusing to classify such content.
6. **Zero-shot strategies**: We were surprised to find that zero-shot strategies worked well compared to chain-of-thought approaches. This suggests a promising direction for future research in exploiting LLMs' ability to learn from fewer examples and still achieve good performance.
7. **Data quality**: Our experiments highlighted the importance of having high-quality data for training and evaluating LLMs. Tweets that seemed misclassified may be due to the inherent complexity of human sentiment and humor, making it challenging for LLMs to classify correctly, especially when they are not fine-tuned for a specific dataset.

#### Reflections
1. **Complexity of Prompt Engineering**: Designing effective prompts requires careful consideration of various factors such as the task at hand, the model's capabilities, and the quality of the data. This complexity underscores the importance of understanding LLMs and their limitations when working with them.
2. **Need for domain-specific fine-tuning**: The performance of LLMs on specific tasks can be improved by fine-tuning them on relevant datasets tailored to the task at hand. For example, a model fine-tuned on sentiment analysis of tweets would likely outperform a general-purpose model in this context.
3. **Ethical considerations**: As LLMs become more powerful and capable of generating human-like text, it is crucial to address ethical concerns such as censorship, privacy, and the potential misuse of these models. Researchers and practitioners must take responsibility for ensuring that AI systems are developed and used responsibly and ethically.
4. **Collaboration and sharing**: Collaboration between researchers, developers, and organizations is essential to advance the field of LLMs and Prompt Engineering. Sharing knowledge, tools, and resources can help accelerate progress and drive innovation in this rapidly evolving area.

#### Recommendations for Future Research
1. **Investigate other parameter settings**: Explore various combinations of Top K, Top P, and Min P to understand their impact on the LLM's performance and find optimal settings for specific tasks.
2. **Fine-tune models on sentiment analysis datasets**: Fine-tuning LLMs on large-scale sentiment analysis datasets tailored to the task at hand can significantly improve their ability to classify sentiments accurately.
3. **Explore zero-shot strategies further**: Delve deeper into zero-shot strategies and understand how they can be effectively applied to various NLP tasks.
4. **Address ethical concerns**: Investigate ways to address ethical considerations in the development and deployment of LLMs, such as implementing mechanisms for censorship while preserving the model's ability to process complex language structures.
5. **Collaborate and share resources**: Collaborate with other researchers, developers, and organizations to share knowledge, tools, and resources, accelerating progress and driving innovation in the field of LLMs and Prompt Engineering.


## Considerations
Through the three established metas, relevant conclusions were drawn:
1. In Meta 3, we explored various models (Llama3.2:3B, Phi3:3.8B, and Llama3.1:8B), temperatures (0.1, 0.3, and 0.6) and prompting techniques (zero-shot, few-shot, chain-of-thought).
2. We understood that sometimes less text is more, low temperature is superior (in this case), and while the size of the model can help performance (Llama3.1:8B), it's not determinant (Phi3:3.8B).
3. In Meta 2, some adopted models achieved very similar accuracies among themselves: MLP and SVM reached 62%, while Naive Bayes reached 61%. In the current meta, we concluded that not all text representation methods work with any model. We also found that the following combinations are the best: MLP with transformer, SVM and Naive Bayes with TF-IDF.
4. By comparing Meta 2 and Meta 3, it's possible to infer that models trained with the dataset (Meta 2) had better performance than pre-trained general-purpose LLMs (Meta 3). As expected, models from Meta 2 were superior because a model trained for multiple tasks will have more difficulty with subjective problems involving humor and irony.
5. The project allowed us to apply and consolidate theoretical and practical knowledge in sentiment analysis and language modeling, exploring advanced tools and approaches.
