import textwrap
from datasets import load_dataset, concatenate_datasets, Dataset
import numpy as np
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import wandb
from matplotlib import pyplot as plt
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

np.random.seed(42)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def get_stats(y_test, predictions):
  # print("Accuracy:", metrics.accuracy_score(y_test, predictions))
  # print("Precision:", metrics.precision_score(y_test, predictions))
  # print("Recall:", metrics.recall_score(y_test, predictions))
  # print("F1-score:", metrics.f1_score(y_test, predictions))
  return metrics.accuracy_score(y_test, predictions), metrics.precision_score(y_test, predictions), metrics.recall_score(y_test, predictions), metrics.f1_score(y_test, predictions) 

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    # tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    return " ".join(tokens)

def wrap_text(text, width = 100):
  return textwrap.fill(text, width=width)
  
def concat_paper_sections(example):
  return "Abstract:\n\n" + example["abstract"] + "\n\nIntroduction:\n\n" + example["introduction"] + "\n\nConclusion:\n\n" + example["conclusion"]

def create_text_dataset(dataset, srcs=['real', 'chatgpt', 'scigen', 'gpt2', 'galactica']):
    has_real = 'real' in srcs
    count_fake_generators = len(srcs) - 1 if has_real else len(srcs)
    filtered_dataset = dataset.shuffle()

    ds_train = []
    ds_test = []

    if has_real:
        ds_train.append(Dataset.from_dict(filtered_dataset['train'].filter(lambda x: x["src"] == "real")[:2000*count_fake_generators]))
        ds_test.append(Dataset.from_dict(filtered_dataset['test'].filter(lambda x: x["src"] == "real")[:1000*count_fake_generators]))

    srcs.remove('real')

    for src in srcs:
        ds_train.append(Dataset.from_dict(filtered_dataset['train'].filter(lambda x: x["src"] == src)[:]))
        ds_test.append(Dataset.from_dict(filtered_dataset['test'].filter(lambda x: x["src"] == src)[:]))

    ds_train = concatenate_datasets(ds_train)
    ds_test = concatenate_datasets(ds_test)

    X_train = ds_train.map(lambda example: {"text": concat_paper_sections(example)})["text"]
    y_train = ds_train.map(lambda example: {"label": example["label"]})["label"]
    X_test = ds_test.map(lambda example: {"text": concat_paper_sections(example)})["text"]
    y_test = ds_test.map(lambda example: {"label": example["label"]})["label"]

    return X_train, X_test, y_train, y_test

red_color_code = "\033[91m"

def add_red_color(match):
  return f"{red_color_code}{match}\033[0m"

def color_matches(regex, text):
  matches = re.findall(regex, text)
  for match in matches:
    text = re.sub(rf"\b{re.escape(match)}\b", add_red_color(match), text)
  return wrap_text(text)

import seaborn as sns
def plot_coeffcients_importance(lr_coeffcients_df = None, rf_feature_imoprtance = None, train_set = "TRAIN", ngram_text = "1"):
  custom_palette = {True: 'green', False: 'red'}
  if lr_coeffcients_df is not None:
    negative_df = lr_coeffcients_df[lr_coeffcients_df['Importance'] < 0]
    positive_df = lr_coeffcients_df[lr_coeffcients_df['Importance'] >= 0]

    negative_df = negative_df.iloc[(-negative_df['Importance'].abs()).argsort()]
    positive_df = positive_df.iloc[(-positive_df['Importance'].abs()).argsort()]

    top_10_negative_df = negative_df.head(10)
    top_10_positive_df = positive_df.head(10)
    lr_coeffcients_df = lr_coeffcients_df.sort_values(by=['Importance'])
    final_df = pd.concat([top_10_negative_df, top_10_positive_df])
    final_df["real"] = final_df["Importance"] <= 0
    plt.figure(figsize=(7, 9))
    sns.barplot(final_df, x = "Importance", y ="Coeffcient", palette=custom_palette, hue="real", dodge=False)
    plt.annotate("Fake", xy=(0.92,-.05), xycoords="axes fraction",
                        xytext=(5,-5), textcoords="offset points",
                        ha="left", va="top", weight='bold')
    plt.annotate("Real", xy=(-0.01,-.05), xycoords="axes fraction",
                        xytext=(5,-5), textcoords="offset points",
                        ha="left", va="top", weight='bold')
    plt.legend('',frameon=False)
    plt.ylabel(f"{ngram_text}gram Feature")
    plt.xlabel("Value")
    plt.title(f"Logistic Regression {ngram_text}gram Features (TF-IDF) ({train_set})")
    plt.show()
  if rf_feature_imoprtance is not None:
    plt.figure(figsize=(7, 9))
    # sns.set_color_codes("pastel")
    sns.barplot(rf_feature_imoprtance[:20], x = "Importance", y ="Feature", color = 'b')
    plt.title(f"Random Forest {ngram_text}gram Features (TF-IDF) ({train_set})")
    plt.ylabel(f"{ngram_text}gram Feature")
    plt.show()

def get_papers_with_text(X_array, label_array, src_array, regex_pattern, print_matches = False, number_of_prints = 0):
  fake_matching_entries = []
  real_matching_entries = []
  # print(regex_pattern)
  # regex_pattern = re.escape(regex_pattern)
  # print(regex_pattern)
  real = 0
  fake = 0
  avg_fake_occ = 0
  avg_real_occ = 0
  src_fake = {"scigen": 0, "chatgpt":0, "gpt2":0, "galactica": 0}
  for i, text in tqdm(enumerate(X_array), total=len(X_array)):
    matches = re.findall(regex_pattern, text)
    label = label_array[i]
    src = src_array[i]
    if len(matches) > 0:
      colorized_text = text
      if label == 0:
        real+=1
        real_matching_entries.append(text)
        avg_real_occ+= len(matches)
      else:
        fake+=1
        src_fake[src] = src_fake[src] + 1
        fake_matching_entries.append(text)
        avg_fake_occ+= len(matches)
      for match in matches:
        if print_matches:
          colorized_text = re.sub(rf"\b{re.escape(match)}\b", add_red_color(match), colorized_text)
          print(wrap_text(colorized_text))
          print(f"label: {label_array[i]}")
          print(f"matched {len(matches)} times")
  real_div = 1 if real==0 else real
  fake_div = 1 if fake==0 else fake
  print(f"found {real} with label 0 (real) with avg. occ. of {avg_real_occ/real_div} per doc and {fake} with label 1 (fake) with avg. occ. of {avg_fake_occ/fake_div} per papr with a count of sources {src_fake} for the {regex_pattern} pattern.")

  for i in range(number_of_prints):
    print("FAKE MATCH ", i)
    print(color_matches(regex_pattern, fake_matching_entries[i]))
  for i in range(number_of_prints):
    print("REAL MATCH ", i)
    print(color_matches(regex_pattern, real_matching_entries[i]))

def get_unique_values(arr):
    unique_values = set(arr)
    return list(unique_values)

def get_feature_importance_rf_lr(X_train, 
                                X_test, 
                                y_train, 
                                y_test, 
                                split_column_vals, 
                                ngram = (1, 1), 
                                tfidf = False, 
                                analyzer = "word", 
                                input_name = "", 
                                lr_model = None, 
                                rf_model = None, 
                                vectorizer = None, 
                                train_set_name = ""):
  wandb.init(project = "DMGSP", name = f"LR-RF-final-stats-ngram={ngram}-tfidf={tfidf}-analyzer={analyzer}-{input_name}")
  if vectorizer is None:
    if tfidf:
      vectorizer = TfidfVectorizer(ngram_range=ngram, analyzer = analyzer)
    else:
      vectorizer = CountVectorizer(ngram_range=ngram, analyzer = analyzer)
    X_train = vectorizer.fit_transform([preprocess_text(text) for text in X_train])

  X_test = vectorizer.transform([preprocess_text(text) for text in X_test])
  column_names = list(vectorizer.get_feature_names_out())
  # print(column_names[0:10])

  ngram_text = (1, 2) if ngram[0] == 1 and ngram[1] == 2 else ngram[0]
  ## RF
  if rf_model is not None:
    rf_classifier = rf_model
  else:
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)
    importances = rf_classifier.feature_importances_
    # wandb.sklearn.plot_feature_importances(rf_classifier, column_names)

    forest_importances = pd.DataFrame({'Importance': importances, 'Feature': column_names})
    forest_importances = forest_importances.sort_values(by = 'Importance', ascending=False)
    RF_table = wandb.Table(dataframe=forest_importances)
    wandb.log({"RF_Feature_Importance" : wandb.plot.bar(RF_table, "Feature", "Importance",
                                title="RF_Feature_Importance")})
    plot_coeffcients_importance(rf_feature_imoprtance=forest_importances, ngram_text=ngram_text, train_set = train_set_name)
  preds_rf = rf_classifier.predict(X_test)
  # print("RF stats:")
  rf_acc, prec, recall, f1 = get_stats(y_test, preds_rf)
  wandb.log({'RF_accuracy': rf_acc, 'RF_precision': prec, 'RF_recall': recall, 'RF_f1': f1})
  for val in get_unique_values(split_column_vals):
    indicies = []
    for i, instance in enumerate(split_column_vals):
      if instance == val:
        indicies.append(i)
    y_test_for_val = [y_test[idx] for idx in indicies]
    y_preds_for_val = [preds_rf[idx] for idx in indicies]
    rf_acc, prec, recall, f1 = get_stats(y_test_for_val, y_preds_for_val)
    wandb.log({f'RF_{val}_accuracy': rf_acc, f'RF_{val}_precision': prec, f'RF_{val}_recall': recall, f'RF_{val}_f1': f1})
  # print(forest_importances[0:20])

  ## LR
  if lr_model is not None:
    regressor = lr_model
  else:
    regressor = LogisticRegression()
    regressor.fit(X_train, y_train)
    coefficients = regressor.coef_[0]
    feature_importance = pd.DataFrame({'Importance': coefficients, 'Coeffcient': column_names})
    feature_importance = feature_importance.reindex(feature_importance['Importance'].abs().sort_values(ascending=False).index)

    top_features = feature_importance#.head(20)
    # print(top_features)
    LR_table = wandb.Table(dataframe=top_features)
    wandb.log({"LR_Coeffcients_Importance" : wandb.plot.bar(LR_table, "Coeffcient", "Importance",
                                title="LR Coeffcients Importance")})
    plot_coeffcients_importance(lr_coeffcients_df=top_features, ngram_text=ngram_text, train_set = train_set_name)
  preds_lr = regressor.predict(X_test)
  # print("LR stats:")
  lr_acc, prec, recall, f1 = get_stats(y_test, preds_lr)
  wandb.log({'LR_accuracy': lr_acc, 'LR_precision': prec, 'LR_recall': recall, 'LR_f1': f1})
  for val in get_unique_values(split_column_vals):
    indicies = []
    for i, instance in enumerate(split_column_vals):
      if instance == val:
        indicies.append(i)
    y_test_for_val = [y_test[idx] for idx in indicies]
    y_preds_for_val = [preds_lr[idx] for idx in indicies]
    acc, prec, recall, f1 = get_stats(y_test_for_val, y_preds_for_val)
    wandb.log({f'LR_{val}_accuracy': acc, f'LR_{val}_precision': prec, f'LR_{val}_recall': recall, f'LR_{val}_f1': f1})
  wandb.finish()
  return lr_acc, rf_acc, regressor, rf_classifier, vectorizer
  

def replace_character_regex(arr, old_char, new_char):
    new_arr = []
    
    for value in arr:
        modified_value = re.sub(old_char, new_char, value)
        new_arr.append(modified_value)
    
    return new_arr

hf_dataset_paper_dataset = {
  "classifier_input_train":"TRAIN",
  "classifier_input_test":"TEST",
  "train+gpt3_train": "TRAIN+GPT3",
  "train-cg_train": "TRAIN-CG",
  "ood_gpt3_test":"OOD-GPT3",
  "ood_real_test":"OOD-REAL",
  "tecg_test":"TECG",
  "test-cc_test":"TEST-CC",
}

def color_based_on_out_of_distribution(train_set_name, test_set_name, accuracy):
  accuracy = "{0:0.1f}".format(accuracy)
  colored_acc_str = f"\cellcolor{{blue!15}}{{{accuracy}\%}}"
  normal_acc = f"{accuracy}\%"
  if test_set_name == "classifier_input_test":
    return normal_acc
  elif test_set_name == "ood_gpt3_test":
    if train_set_name == "train+gpt3_train":
      return normal_acc
    else:
      return colored_acc_str
  elif test_set_name == "tecg_test":
    if train_set_name == "train-cg_train":
      return colored_acc_str
    else:
      return normal_acc
  else:
    return colored_acc_str


def run_model_on_datasets(remove_ligature_letters = True, ngram = (1, 1), tfidf = True, analyzer = 'word',
                          included_datasets = ['classifier_input', 'ood_gpt3', 'ood_real', 'tecg', 'test-cc', 'train+gpt3', 'train-cg'], ignore_tests = [], split_column_dict = {"test-cc": "paraphrased_sections"}):
  results_df_lr = pd.DataFrame()
  results_df_rf = pd.DataFrame()
  train_dataset_names = []
  latex_str_lr = ""
  latex_str_rf = ""
  for dataset_train_name in included_datasets:
    dataset = load_dataset('tum-nlp/IDMGSP', dataset_train_name)
    dataset_parsed = dataset.map(lambda example: {"text": concat_paper_sections(example)})
    # if the dataset has a train dataset.
    if "train" in dataset:
      if dataset_train_name == "ood_gpt3":
        continue
      curr_lr_model = None
      curr_rf_model = None
      curr_vectorizer = None
      train_set_name = dataset_train_name + "_train"
      ngram_text = (1, 2) if ngram[0] == 1 and ngram[1] == 2 else ngram[0]
      latex_str_lr+= f"LR-{ngram_text}gram (tf-idf) & {hf_dataset_paper_dataset[train_set_name]} &"
      latex_str_rf+= f"RF-{ngram_text}gram (tf-idf) & {hf_dataset_paper_dataset[train_set_name]} &"
      X_train = dataset_parsed["train"]["text"]
      if remove_ligature_letters:
        X_train = [remove_ligatures(text) for text in X_train]
      y_train = dataset_parsed["train"]["label"]
      src_train = dataset_parsed["train"]["src"]
      train_dataset_names.append(dataset_train_name + "_train")
      for dataset_test_name in included_datasets:
        if dataset_test_name in ignore_tests:
            continue
        dataset = load_dataset('tum-nlp/IDMGSP', dataset_test_name)
        dataset_parsed = dataset.map(lambda example: {"text": concat_paper_sections(example)})
        # if the dataset has a test dataset.
        if "test" in dataset:
          test_set_name = dataset_test_name + "_test"
          X_test = dataset_parsed["test"]["text"]
          if remove_ligature_letters:
            X_test = [remove_ligatures(text) for text in X_test]
          y_test = dataset_parsed["test"]["label"]
          split_column_name = split_column_dict[dataset_test_name] if dataset_test_name in split_column_dict else "src"
          split_column = dataset_parsed["test"][split_column_name]
          lr_acc, rf_acc, out_lr_model, out_rf_model, out_vectorizer = get_feature_importance_rf_lr(
                                                                                    X_train, 
                                                                                    X_test, 
                                                                                    y_train, 
                                                                                    y_test, 
                                                                                    split_column, 
                                                                                    ngram=ngram, 
                                                                                    tfidf=tfidf, 
                                                                                    analyzer=analyzer, 
                                                                                    input_name = f"remove_ligature_letters={remove_ligature_letters}-{dataset_train_name}_train-{dataset_test_name}_test",
                                                                                    lr_model=curr_lr_model,
                                                                                    rf_model=curr_rf_model,
                                                                                    vectorizer=curr_vectorizer,
                                                                                    train_set_name = hf_dataset_paper_dataset[train_set_name])
          curr_lr_model = out_lr_model
          curr_rf_model = out_rf_model
          curr_vectorizer = out_vectorizer
          results_df_lr.loc[train_set_name, test_set_name] = lr_acc
          results_df_rf.loc[train_set_name, test_set_name] = rf_acc
          latex_str_lr+= f" {color_based_on_out_of_distribution(train_set_name, test_set_name, lr_acc*100)} &"
          latex_str_rf+= f" {color_based_on_out_of_distribution(train_set_name, test_set_name, rf_acc*100)} &"
      latex_str_lr = latex_str_lr[:-1] + "\\\\\n"
      latex_str_rf = latex_str_rf[:-1] + "\\\\\n"
  print(latex_str_lr)
  print(latex_str_rf)
  wandb.init(project = "DMGSP", name = f"LR-RF-final-acc-complete-ngram={ngram}-tfidf={tfidf}-analyzer={analyzer}-remove_ligature_letters={remove_ligature_letters}")
  results_df_lr["training_dataset"] = train_dataset_names
  # reorder
  results_df_lr.insert(0, 'training_dataset', results_df_lr.pop('training_dataset'))
  results_df_rf["training_dataset"] = train_dataset_names
  # reorder
  results_df_rf.insert(0, 'training_dataset', results_df_rf.pop('training_dataset'))
  LR_table = wandb.Table(dataframe=results_df_lr)
  RF_table = wandb.Table(dataframe=results_df_rf)
  wandb.log({'LR_Accuracy': LR_table})
  wandb.log({'RF_Accuracy': RF_table})
  wandb.finish()
  print(results_df_lr)
  print(results_df_rf)

ligature_mapping = {
    'ﬀ': 'ff',
    'ﬁ': 'fi',
    'ﬂ': 'fl',
    'ﬃ': 'ffi',
    'ﬄ': 'ffl',
    # Add more ligature mappings as needed
}

# Function to remove ligature letters by replacing them with actual letters
def remove_ligatures(text):
    for ligature, replacement in ligature_mapping.items():
        text = text.replace(ligature, replacement)
    return text