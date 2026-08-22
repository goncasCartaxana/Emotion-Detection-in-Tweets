# Meta 1: Annotations and Data Analysis
A sample of Tweets was manually annotated with sentiment scores ranging from [-2, 1]. The dataset used for this project is a portion of the TwitterDialogueSAPT dataset (Carvalho et al., 2022), which consists of dialogues in Portuguese gathered from Twitter. The data includes interactions related to e-commerce services, television, and healthcare.

Our specific subset involved dialogues related to Netflix (@NetflixPT), RTP Notícias (@RTPNoticias), and Vodafone (@VodafonePT). The corpus contains informal conversations with abbreviations, internet slang, and English expressions.

## Annotations
The annotation process was individual. Emerging criteria for annotators included:
- All humorous phrases tend to be positive.
- A distinction between negative (-2 and -1) categories is related to the perceived level of user dissatisfaction or aggressive tone of vocabulary used.
- Minimal sadness or enthusiasm is considered positive;
- The neutral (0) value usually indicates that the message is merely informative.

## Data Analysis
1. Data cleaning removed NaN values in `dialog_ID`.
2. Two popular NLP libraries were used for corpus analysis (NLTK and spaCy).
3. Stopwords and punctuation were eliminated, and the total number of tokens and of unique tokens were determined using both tools.
4. NLTK identified common tokens, while spaCy offered functions for entity and link identification.

## Agreement Analysis
The statistical metrics used to evaluate inter-annotator agreement were Krippendorff's alpha, Cohen's Kappa, and Fleiss' Kappa.

### Annotator Consistency
The calculated metrics revealed a disparity between the indices:
- According to the Kappa coefficients, the values indicate a substantial level of agreement.
- However, Krippendorff's alpha indicated a low level of consistency.
This divergence suggests the metrics responded differently to the data distribution, with Alpha penalizing disagreements or category imbalances more rigorously than the Kappa variants.

### Entity-Level Agreement and Sentiment Analysis
An analysis of the responses by entity revealed distinct sentiment trends:
- The majority of the dataset consists of responses related to Netflix, which primarily garnered neutral sentiments.
- In contrast, Vodafone and RTP received fewer responses, both leaning negative.
	- Vodafone exhibited a higher concentration of extremely negative (-2) sentiments, driven by user dissatisfaction with the service provided.
	- RTP leaned toward slightly negative (-1) sentiments, which can be attributed to sensitive content regarding war. 

## Discussion
The difference in sentiments among entities: Vodafone received a high volume of extremely negative responses, whereas Netflix was generally categorized as neutral. RTP received slightly negative responses. 
The reasons include the content of tweets from the entities and dissatisfaction with the services provided (particularly from Vodafone).

## Limitations
The limited size of the dataset (100 Tweets) may restrict the generalization of results for a broader analysis of sentiments in digital interactions.
