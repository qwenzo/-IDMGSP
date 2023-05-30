# GPT-2 generators

## This directory contains the notebooks for fine-tuning GPT-2 from [Hugging Face](https://huggingface.co/)

1. Generate `train_gpt2.csv` and `val_gpt2.csv` from the dataset which contains the scaped real papers [here](https://gitlab.lrz.de/lab-courses/xai-lab-ws22-23/teams-edoardo/uncover-artificially-generated-text/-/tree/main/data).
2. Run `from_csv_to_txt.py` to prepare the input files which should be in .txt form.
3. Generate abstracts from titles using `GPT2_from_title_to_abstract.ipynb` and uploading `train_gpt2_abstract.txt`, `val_gpt2_abstract.txt` and the file contains the list of titles, from which you would like to start generating, in the same directory of the notebook.
4. Generate introductions from titles using `GPT2_from_title_to_introduction.ipynb` and uploading `train_gpt2_introduction.txt`, `val_gpt2_introducion.txt` and the file contains the list of titles, from which you would like to start generating, in the same directory of the notebook.
5. Generate conclusions from titles using `GPT2_from_title_to_conclusion.ipynb` and uploading `train_gpt2_conclusion.txt`, `val_gpt2_conclusion.txt` and the file contains the list of titles, from which you would like to start generating, in the same directory of the notebook.

The datasets used for the training and generation procedures can be found [here](https://gitlab.lrz.de/lab-courses/xai-lab-ws22-23/teams-edoardo/uncover-artificially-generated-text/-/tree/main/data/papers_csv).

Moreover, in the directory is present the file `merge_csv.py`, useful for unifying the results.
