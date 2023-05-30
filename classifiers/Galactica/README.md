# Galactica Classifier
## Files and Folders
- `results`  contains the output models.
    - `82-trainedOnNewData` model trained on TRAIN.
    - `84-noChatGPT` model trained on TRAIN-CG.
    - `85-TrainedOnGPT3` model trained on TRAIN+GPT3.
- `explainability.ipynb` explainability notebook.
- `galactica_classification.ipynb` classification notebook.
- `INCOMPLETEHuggingFaceIntegration.ipynb` incomplete notebook for pushing the model to huggingface.
- `galactica_complete_preds.csv` csv file containing prediction, label, src and softmax probability of `82-trainedOnNewData` model tested on TEST dataset.