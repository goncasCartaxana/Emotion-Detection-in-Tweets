
# Meta 3: Prompt Engineering with LLMs
In this final meta, we explore Large Language Models (LLMs), focusing on Prompt Engineering techniques. 
The goal is to understand how LLMs can be used to classify sentiment by crafting appropriate prompts that guide their reasoning process.

## Overview of LLMs
Large Language Models are powerful AI models that can process a vast amount of textual data, with the processing time and computational complexity increasing non-linearly (usually $O(n^2)$). 
In this project, a system prompt containing too many tokens was observed to significantly increase the processing time.

## Prompting
Prompt Engineering involves designing prompts to guide LLMs in performing specific tasks. 

Here are some benefits of using a chain-of-thought (CoT) approach:
* Improved performance (accuracy)
* Enhanced interpretability (a layed out reasoning process, leading to more transparency)
* Better generalization (helps with complex tasks)

We will focus on the following factors:
1. LLM (choosing a model of up to 8B parameters):
	* Models
	* Temperature
2. System Prompt:
	* Description: Defining the problem (Role)
	* Context: Additional information that aids in the task (Criteria)
	* Format: How we ask for the response (Chain-of-Thought + sentiment retrieval)
3. Capturing sentiment value using Regex
4. Iterating up to three times when Regex fails to identify a valid value

## Sets of Experiments

### Experiment 1 - Baseline
We started by testing simple prompts with different structures, aiming to explain the LLM's objective: classify the sentiment of tweets. 
The best result was obtained using the baseline 1 prompt; therefore, we decided to use it for the subsequent experiments. This was using a temperature of 0.1.

### Experiment 2 - Temperature
We tested three new temperature values (0.2, 0.3, and 0.6) regarding temperature. 
A temperature of 0.3 was closest to the previous obtained accuracy. However, the temperature of 0.1 still performed better.

### Experiment 3 - Criteria
With CoT and context, we found that shorter and more objective phrases resulted in improved performance, leading to an increase in accuracy by 11.25%. 
This improvement marked the best result achieved so far with a 50% accuracy rate. 
a new "devil advocate" approach failed to yield any significant improvement, as the LLM's creativity did not positively impact the performance.

### Experiment 4 - Models (LLMs Larger)
We concluded that the quality of the system prompt is most crucial in determining the model's response quality. 

The "baseline 1" prompt was chosen to test  larger models, even though it presented a slight reduction in accuracy compared to "COT + context", because of the processing efficiency gain noticed due to the amount of text in the prompt.


## Discussion
Tweets where the models refused to classify were noticed. Remedies for this issue could be:
1. Preprocessing to replace insults with generic terms (e.g., insult) or
2. Using an uncensored model.

Examples:
* Tweet(censored): "Dados moveis da @user com defeito, p@t@ m3rd@"
* Answered with: "Desculpe, mas não posso cumprir esse pedido."

Strategies with better results:
* Surprisingly, the zero-shot strategy worked well compared to the strategies using chain-of-thought.
* A temperature of 0.1 performed better than expected even with chain-of-thought.

Performance depends on the quality of the data:
- We found examples where tweets seemed misclassified. For instance, the following tweets should have been classified as 1:
- "@user Muito obrigada, @user. ❤ " is classified as -2.
- "@user finalmente ♥️ " is classified as 0.
- There are tweets identical with different classifications. Example: "Bolo rei é péssimo! 😠 " is classified both as -1 and -2.

## Considerations
In this final meta, we explored various aspects of LLMs and Prompt Engineering to classify sentiment in tweets. 
Through our experiments, we gained valuable insights and made some general conclusions, reflections, and recommendations for future research:

### General Conclusions
1. **Quality of the system prompt**: Our results demonstrated that the quality of the system prompt significantly affects the LLM's performance. A well-crafted prompt helps guide the model towards the desired task and improves its accuracy.
2. **Temperature settings**: We discovered that a lower temperature value generally led to better performance, although there were exceptions where higher temperatures worked better. Future research could explore more fine-tuning of temperature values for specific tasks.
3. **The role of context**: Although we did not explicitly use DialogID during our experiments, providing better context to the LLM could potentially help improve its ability to classify sentiments accurately.
4. **Exploring other parameters**: We only explored a few parameters such as temperature and prompt format; future research should delve into other parameters like Top K, Top P, and Min P that determine which tokens are sampled during processing.
5. **Preprocessing and censorship**: When dealing with tweets containing offensive language or insults, pre-processing to replace these terms with generic ones or using an uncensored model could help address the issue of LLMs refusing to classify such content.
6. **Zero-shot strategies**: We were surprised to find that zero-shot strategies worked well compared to chain-of-thought approaches. This suggests a promising direction for future research in exploiting LLMs' ability to learn from fewer examples and still achieve good performance.
7. **Data quality**: Our experiments highlighted the importance of having high-quality data for training and evaluating LLMs. Tweets that seemed misclassified may be due to the inherent complexity of human sentiment and humor, making it challenging for LLMs to classify correctly, especially when they are not fine-tuned for a specific dataset.

### Reflections
1. **Complexity of Prompt Engineering**: Designing effective prompts requires careful consideration of various factors such as the task at hand, the model's capabilities, and the quality of the data. This complexity underscores the importance of understanding LLMs and their limitations when working with them.
2. **Need for domain-specific fine-tuning**: The performance of LLMs on specific tasks can be improved by fine-tuning them on relevant datasets tailored to the task at hand. For example, a model fine-tuned on sentiment analysis of tweets would likely outperform a general-purpose model in this context.
3. **Ethical considerations**: As LLMs become more powerful and capable of generating human-like text, it is crucial to address ethical concerns such as censorship, privacy, and the potential misuse of these models. Researchers and practitioners must take responsibility for ensuring that AI systems are developed and used responsibly and ethically.
4. **Collaboration and sharing**: Collaboration between researchers, developers, and organizations is essential to advance the field of LLMs and Prompt Engineering. Sharing knowledge, tools, and resources can help accelerate progress and drive innovation in this rapidly evolving area.

## Recommendations for Future Research
1. **Investigate other parameter settings**: Explore various combinations of Top K, Top P, and Min P - how tokens are sampled - to understand their impact on the LLM's performance and find optimal settings for specific tasks.
2. **Fine-tune models on sentiment analysis datasets**: Fine-tuning LLMs on large-scale sentiment analysis datasets tailored to the task at hand can improve their ability to classify sentiments accurately.
3. **Explore zero-shot strategies further**: Delve deeper into zero-shot strategies and understand how they can be effectively applied to various NLP tasks.
4. **Give context in prompts**: Experiment how to provide better context to the LLM so it can help classify sentiments more accurately.
5. **Quantization**: Compare how different model quantizations affect its performance, and if the gain in efficiency compensates the potential loss of performance.
