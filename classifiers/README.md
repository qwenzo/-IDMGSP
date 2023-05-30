# Classifiers
## Files and Folders
- `data` 
    - `stats and cleaning.ipynb` notebook for filtering the data and showing statistics.
    - `archive` older data. Not used for classification.
    - `data_nlp2022` data of Shrestha and Zhou (2022).
        - `gpt3curienlp2022.csv` GPT-3 curie data taken from Shrestha and Zhou (2022).
        - `gpt3curienlp2022_restricted.csv` filtered GPT-3 curie data taken from Shrestha and Zhou (2022).
        - `gpt3curienlp2022_restricted_test.csv` 1k papers taken from `gpt3curienlp2022_restricted.csv`. Used for testing the classifiers.
        - `gpt3curienlp2022_restricted_train.csv` 1.2k papers taken from `gpt3curienlp2022_restricted.csv`. Used for training the classifiers (OOD-GPT3).
        - `realnlp2022.csv` real data taken from Shrestha and Zhou (2022).
        - `realnlp2022_restricted.csv` filtered real data taken from Shrestha and Zhou (2022).
        - `realnlp2022_restricted_4000.csv` filtered and randomly selected 4k real data taken from Shrestha and Zhou (2022) (OOD-REAL).
    - `classifier_input_restricted.csv` complete filtered dataset for classification.
    - `classifier_input_restricted_train.csv` complete filtered dataset for training the classifiers (TRAIN).
    - `classifier_input_restricted.csv` complete filtered dataset for testing the classifiers (TEST).
- `Galactica` contains output models and classification notebook for Galactica.
- `GPT3` contains classification notebook for GPT3.
- `Galactica` contains classification notebook for RoBERTa.