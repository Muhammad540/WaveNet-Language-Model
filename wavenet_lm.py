import marimo

__generated_with = "0.11.13"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        """
        # Wavenet Inspired LM

        We're going to train a character-level autoregressive language model using a large dataset of company names I found online. Then, we'll sample from the model to generate new company names for our imaginary startup :)

        [Read the WaveNet paper ~ Google Deepmind](https://deepmind.google/discover/blog/wavenet-a-generative-model-for-raw-audio/)
        """
    )
    return


@app.cell
def _():
    import torch 
    import torch.nn.functional as F 
    import csv  
    import requests  
    import re
    return F, csv, re, requests, torch


@app.cell
def _(mo):
    mo.md(r"""### Scrape the data""")
    return


@app.cell
def _(csv, re, requests):
    url = "https://raw.githubusercontent.com/prasertcbs/basic-dataset/refs/heads/master/companies.csv"  
    response = requests.get(url)  
    response.raise_for_status()  
    csv_data = response.content.decode('utf-8').splitlines()  
    names = []  
    reader = csv.DictReader(csv_data)  

    # Fix: Define characters to remove properly  
    chars_to_remove = " {}\'!#&(),-./:\"0123456789\\"  

    for row in reader:  
        # Get the original name  
        original_name = row['name_latest']  
    
        # Remove unwanted characters using regex  
        clean_name = re.sub(f'[{re.escape(chars_to_remove)}]', '', original_name)  
    
        names.append(clean_name)  

    print(f"Total company names found {len(names)} !")  
    print(f"Longest name with {max(len(name) for name in names)} characters !")  
    print(f"Some samples: {names[:5]}")  
    return (
        chars_to_remove,
        clean_name,
        csv_data,
        names,
        original_name,
        reader,
        response,
        row,
        url,
    )


@app.cell
def _(mo):
    mo.md(
        """
        ### First step 
        1. Build the vocabulary for your model
        2. Map the characters to integers and vice versa

        This is kind of tokenization
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(names):
    # get all possible chars in the dataset (52 = 26 alphabets * 2)
    chars = sorted(list(set(''.join(names))))
    string_to_integer = {s:i+1 for i,s in enumerate(chars)} # s:i+1 since we'll reserve the 0th integer to be '.' dot or EOS (end of sequence token)
    string_to_integer['.'] = 0
    integer_to_string = {i:s for s,i in string_to_integer.items()}
    vocabulary_size = len(integer_to_string)
    print(f" Integer to string mapping: {integer_to_string}")
    print(f" String to integer mapping: {string_to_integer}")
    return chars, integer_to_string, string_to_integer, vocabulary_size


@app.cell
def _(mo):
    context_slider = mo.ui.slider(start=1, stop=20, step=2)
    context_slider
    return (context_slider,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Build the dataset

        Context length is the number of characters that your model will see when it tries to predict the next character.
        We will use each name as a training example. So say the company name is "Google", we will create training examples in following manner, 

         X(input) -> 🧠 -> Y(output)
   
        Training example 1

        . . . -> G

        Training example 2

        . . G -> o

        Training example 3

        . G o -> o

        Training example 4

        G o o -> g

        Training example 5


        o o g -> l

        Training example 6

        o g l -> e

        Training example 8

        g l e -> .
        """
    )
    return


@app.cell
def _(context_slider, string_to_integer, torch):
    context_length = context_slider.value
    def build_dataset(names):
        # there is always X -> Y mapping 
        X, Y = [], []
        for name in names:
            # remeber that we reserved 0 as '.'
            context = [0] * context_length
            # each name must end with a '.' -> our model will learn this behaviour
            for character in name + '.':
                index = string_to_integer[character]
                X.append(context)
                Y.append(index)
                context = context[1:] + [index]
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        print(f"Dataset shape: {X.shape, Y.shape}")
        return X,Y
    return build_dataset, context_length


@app.cell
def _(mo):
    mo.md("""#### chunk the dataset into train, val and test set""")
    return


@app.cell
def _(build_dataset, names):
    split_percentage_1 = int(0.8*len(names))
    split_percentage_2 = int(0.9*len(names))
    # first 80 percent goes to training
    Xtrain,Ytrain = build_dataset(names[:split_percentage_1])
    # next 10 percent goes to validation
    Xval,Yval = build_dataset(names[split_percentage_1:split_percentage_2])
    # last 10 percent goes to test
    Xtest,Ytest = build_dataset(names[split_percentage_2:])
    return (
        Xtest,
        Xtrain,
        Xval,
        Ytest,
        Ytrain,
        Yval,
        split_percentage_1,
        split_percentage_2,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
