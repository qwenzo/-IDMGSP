# Classifiers
## Files and Folders
- `data` 
    - `stats and cleaning.ipynb` notebook for filtering the data and showing statistics.
    - `archive` older data. Not used for classification.
    - `data_nlp2022` OOD datasets
        - `gpt3curienlp2022.csv` OOD-GPT3.
        - `gpt3curienlp2022_restricted.csv` filtered OOD-GPT3.
        - `gpt3curienlp2022_restricted_test.csv` 1k papers taken from `gpt3curienlp2022_restricted.csv`. Used for testing the classifiers.
        - `gpt3curienlp2022_restricted_train.csv` 1.2k papers taken from `gpt3curienlp2022_restricted.csv`. Used for training the classifiers (OOD-GPT3).
        - `realnlp2022.csv` real data generated independent of classifier_input_restricted.csv.
        - `realnlp2022_restricted.csv` filtered realnlp2022.csv.
        - `realnlp2022_restricted_4000.csv` randomly selected 4k real data from realnlp2022_restricted.csv (OOD-REAL).
    - `classifier_input_restricted.csv` complete filtered dataset for classification.
    - `classifier_input_restricted_train.csv` complete filtered dataset for training the classifiers (TRAIN).
    - `classifier_input_restricted.csv` complete filtered dataset for testing the classifiers (TEST).
- `Galactica` contains output models and classification notebook for Galactica.
- `GPT3` contains classification notebook for GPT3.
- `RoBERTa` contains classification notebook for RoBERTa.
