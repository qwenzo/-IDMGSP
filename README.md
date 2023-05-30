# Distinguishing Fact from Fiction: A Benchmark Dataset for Identifying Machine-Generated Scientific Papers in the LLM Era.

As generative NLP can now produce content nearly indistinguishable from human writing, it becomes difficult to identify genuine research contributions in academic writing and scientific publications. Moreover, information in NLP-generated text can potentially be factually wrong or even entirely fabricated. This study introduces a novel benchmark dataset, containing human-written and machine-generated scientific papers from SCIgen, GPT-2, ChatGPT, and Galactica. After describing the generation and extraction pipelines, we also experiment with three distinct classifiers as a baseline for detecting the authorship of scientific text. A strong focus is put on generalization capabilities and explainability to highlight the strengths and weaknesses of detectors. We believe our work serves as an important step towards creating more robust methods for distinguishing between human-written and machine-generated scientific papers, ultimately ensuring the integrity of scientific literature.

## Dataset
https://huggingface.co/datasets/tum-nlp/IDMGSP

## Files
- `classifiers` contains code and data used for classification. 
- `data` contains code for scraping real papers and the data used to generate fake ones. 
- `generators` contains code for generators.
