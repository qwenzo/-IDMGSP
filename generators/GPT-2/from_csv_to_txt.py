import pandas as pd

# files paths
filenames = [
    "train_gpt2.csv",
    "val_gpt2.csv",
    "data/papers_csv/test_gpt2.csv",
]

# paper's extracted parts
arguments = ["abstract", "introduction", "conclusion"]
start_token = "<|start|>"
sep_token = "<|sep|>"
end_token = "<|end|>"

# for the train and validation set
for filename in filenames[:-1]:
    # open the correspondent .csv file
    df = pd.read_csv(filename)
    # for every registered part of the papers
    for argument in arguments:
        # create a new file .txt
        with open(filename[:-4] + "_" + argument + ".txt", "w") as f1:
            # append the reformatted text completed with the dividers
            for (
                index,
                row,
            ) in df.iterrows():
                f1.write(
                    # start_token +
                    row["title"]
                    + sep_token
                    + str(row[argument])
                    # + end_token
                    + "\n"
                )

# for the test set
# open the correspondent .csv file
# df = pd.read_csv(filenames[-1])
# create a new file .txt
# with open(filenames[-1][:-4] + ".txt", "w") as f1:
# append the prompts for generation
#    for index, row in df.iterrows():
#        f1.write(start_token + row["real title"] + sep_token + "\n")
