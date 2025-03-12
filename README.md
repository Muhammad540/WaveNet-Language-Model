# WaveNet Language Model

This is an auto-regressive character-level model inspired by the **WaveNet** architecture. We group tokens together sequentially to predict the next token. The model is trained on a large dataset of company names, allowing it to generate new company names for our imaginary startup.

## Getting Started

I built this project using pure PyTorch. You'll notice that we write our own layers, which is inspired by how PyTorch implements them—great for learning!

I recommend going over the 'WaveNet' and 'Neural Probabilistic Language Model' papers for a deeper understanding of the concepts behind this model.

To run the notebook, follow these steps:

1. **Install Marimo**  
   First, install Marimo by visiting the [Marimo website](https://marimo.io/).

2. **Run the Notebook**  
   Execute the following command in your terminal:

   ```bash
   marimo edit wavenet_lm.py
   ``` 
   Open the notebook and execute the cells. Make sure to pay attention to the code and comments throughout.

## Objectives

This project is primarily for learning. I’ve aimed to explain my thought process and reasoning behind the code. You’ll notice a lot of testing in the cells labeled as **Scratchpad**.

### Things to Try

Feel free to experiment with the model’s hyperparameters! Here are some ideas:

- Increase the **Context Window**
- Modify the model architecture by changing or adding layers
- Change the **Embedding Dimension**
- Increase the **Hidden Dimension**
- Switch the dataset to something else
- Tweak the **Learning Rate**, **Batch Size**, and **Number of Epochs**

All these configurations are easy to adjust in the notebook using sliders. Test these out and observe their impact on training, validation, and test loss.