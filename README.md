
# Assessing Thai Dialect Performance in LLMs with Automatic Benchmarks and Human Evaluation (2025)

## Abstract from our paper

Large language models show promising results in various NLP tasks. Despite these successes, the robustness and consistency of LLMs in underrepresented languages remain largely unexplored, especially concerning local dialects. Existing benchmarks also focus on main dialects, neglecting LLMs' ability on local dialect texts. In this paper, we introduce a Thai local dialect benchmark covering Northern (Lanna), Northeastern (Isan), and Southern (Dambro) Thai, evaluating LLMs on five NLP tasks: summarization, question answering, translation, conversation, and food-related tasks. Furthermore, we propose a human evaluation guideline and metric for Thai local dialects to assess generation fluency and dialect-specific accuracy. Results show that LLM performance declines significantly in local Thai dialects compared to standard Thai, with only proprietary models like GPT-4o and Gemini2 demonstrating some fluency

## Results: Traditional Metric



## Citation

N/A


## Run an Eval

### Install
```sh
pip install -r requirements.txt
```

### Run Eval
```sh
bash eval_only.sh
```
