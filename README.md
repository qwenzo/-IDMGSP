# Scifake: Uncovering Machine Generated Scientific Papers in the age of ChatGPT and Co.

State-of-the-art Large Language Models (LLMs) are able to generate factual text based on the data they were trained on. LLMs can also generate completely new information that may not be accurate or entirely fabricated. That raises the question of how to detect if a piece of text is humanly written or produced by a LLM, especially in academia and scientific writing. In this paper, we introduce a dataset of abstracts, introductions, and conclusions extracted from human-written and machine-generated scientific papers based on the arXiv database. We fine-tune neural machine-generated text detectors based on this dataset and employ explainability methods to approximate the most likely features that contribute to the detection process. 
Our models perform exceptionally well when evaluated on in-domain datasets, and with a limited number of out-of-domain examples, they demonstrate the ability to enhance their performance on out-of-domain datasets. Overall, our approach can effectively detect whether a piece of scientific paper is generated by a LLM or written by a human, thereby aiding in the identification and prevention of potentially misleading or false information.

## Files
- `classifiers` contains code and data used for classification. 
- `data` contains code for scraping real papers and the data used to generate fake ones. 
- `generators` contains code for generators.
- `Examples.pdf` contains three genererated papers for each generator. 
- `Lab_Report.pdf` is the final report of the lab. 
- `Midterm_Presentation.pdf` contains the slides used during the midterm presentation.
- `Poster.png` is the poster shown during the final presentation. 