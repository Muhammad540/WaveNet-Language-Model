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
def _(mo):
    mo.md("""### let's take a lot at the architecture and how is it different from a normal MLP ? where are the waves ?""")
    return


@app.cell
def _():
    import torch 
    import torch.nn.functional as F 
    import csv  
    import requests  
    import re
    from typing import Optional
    return F, Optional, csv, re, requests, torch


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
    # scratchpad 
    q = ''.join(names)
    print(q)
    # remove the repetitions 
    w = list(set(q))
    print(w)
    # sort them so we can properly index 
    s = sorted(w)
    print (s)
    return q, s, w


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
    context_slider = mo.ui.slider(start=2, stop=20, step=2)
    context_slider
    return (context_slider,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Build the dataset

        Context length is the number of characters that your model will see when it tries to predict the next character.
        We will use each name as a training example. So say the company name is "Google", we will create training examples in following manner,
        """
    )
    return


@app.cell
def _(mo):
    mo.image(src="trainingexample.png")
    return


@app.cell
def _(context_slider, string_to_integer, torch):
    context_length = context_slider.value
    def build_dataset(names):
        # there is always X -> Y mapping 
        X, Y = [], []
        for name in names:
            # remember that we reserved 0 as '.'
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
def _(mo):
    mo.md(
        r"""
        #### lets see how training data looks like 
        each line represent a training example with inputs and outputs
        """
    )
    return


@app.cell
def _(Xtrain, Ytrain, integer_to_string):
    print("Mapping from X -> Y")
    for x,y in zip(Xtrain[:10],Ytrain[:10]):
        print(''.join((integer_to_string[index.item()] for index in x)), "--->", integer_to_string[y.item()])
    return x, y


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Let's discuss the architecture before we start coding up all the bits and pieces 

        Initial inspiration: A Neural Probabilistic Language Model
        WaveNet Inspiration: WaveNet: A Generative Model for Raw Audio
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        #### Code up the Linear Layer
        We will be taking some inspiration from how pytorch has developed their Linear layers
        [Take a look at nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
        """
    )
    return


@app.cell
def _(torch):
    class Linear:
        '''Applies linear transformation to the input, y = xW + b
            input_features: size of input sample 
            output_featres: size of output sample 
            bias: if set to False, we'll not add the bias 'b'. By default it is True
        '''
        def __init__(self, input_features: torch.Tensor, output_features: torch.Tensor, bias= True):
            # perform xavier initialization
            self.weight = torch.randn((input_features, output_features)) / input_features**0.5
            self.bias = torch.zeros(output_features) if bias else None

        def __call__(self, x:torch.Tensor):
            '''x is your input to this Layer'''
            self.output = x@self.weight
            if self.bias is not None:
                self.output += self.bias
            return self.output

        def parameters(self):
            ''' utility method to return the total number of parameters in this layer'''
            return [self.weight] + ([] if self.bias is None else [self.bias])
    return (Linear,)


@app.cell
def _(mo):
    mo.md(
        r"""
        #### You might be wondering why did we divide the weight by $\sqrt(input features)$ ? 
        _**Lets discuss Xavier initialization briefly:**_

        It is a structured way for initializing the weights of a neural network to improve gradient flow during backprop and avoid vanishing or exploding gradient problems also it leads to faster convergence

        Instead of initializing weights from a standard normal distribution $N(0,1)$, we use a normal distribution with a mean of 0 and a variance that is specifically calculated to maintain the variance of activations **across layers**. This ensures that the variance of the inputs to a layer is approximately equal to the variance of the outputs from that layer.

        The formula for initializing the weights of a particular layer is: 
        $$W \sim N(0, \text{var})$$  
        where 
        $$var = \frac{1}{n}$$
        and n is the number of input neurons to that layer. 
        We know that,
        $$\sigma=\sqrt(variance)$$

        So when we use **torch.randn** it give us a standard normal distribution with mean 0 and variance 1, which we then multiply by the corrected standard deviation from the xavier initialization.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        #### Code up BatchNorm 
        [Take a look at Pytorch impl](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)

        Before we code, lets take a look at the formulation that pytorch describes

        $$y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta$$

        So we see that,

        $$\frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}}$$

        is the normalization term which centers the data around zero mean and scales it to have a unit variance.

        $$\gamma, \beta$$

        are learnable scaling and shifting params that allows the model to understand the optimal shift and scale for the normalized data.
        """
    )
    return


@app.cell
def _(torch):
    # scratch pad
    A = torch.randn((3,4))
    print(A.ndim)
    B = torch.randn((3,4,5))
    print(B.ndim)
    C = torch.randn(3,4)
    print(C)
    #compute batch norm
    cmean = C.mean(dim=0)
    print(cmean,cmean.shape) # oops, look at the shape, what we expected was 1,4 but got 4
    cmean_tryagain = C.mean(dim=0,keepdim=True)
    print(cmean_tryagain,cmean_tryagain.shape) # now it looks nice
    return A, B, C, cmean, cmean_tryagain


@app.cell
def _(Optional, torch):
    class BatchNorm1d:
        ''' compute the batch norm
            num_features: number of features in input
            momentum: to keep the running mean and average
            eps: well avoid divide by zero errors
            training: whether we are in training or evaluation mode ( this layer has different behavior in each )
        '''
        def __init__(self, num_features: int, momentum: Optional[float]=0.1, eps: float = 1e-5, training: bool = True):
            self.eps = eps
            self.momentum = momentum
            self.is_training = training

            self.gamma = torch.ones(num_features)
            self.beta = torch.zeros(num_features)

            self.running_mean = torch.zeros(num_features)
            self.running_var = torch.ones(num_features)

        def __call__(self, x):
            if self.is_training:
                # Heads up ! We are designing our Batch Norm layer to work with tensor of dims 2 or 3 
                # so for 2D case (N,C) we'll compute the norm across the N or '0' dim
                # and for the 3D case (N,L,C) where N is the batch dim, L is the sequence of features, and C is the feature  dim. So we'll               norm across the batch size for all sequences of each and every feature '0,1'
                if x.ndim==2:
                    dim = 0
                elif x.ndim==3:
                    dim = (0,1)
                # batch mean and variance
                xmean = x.mean(dim,keepdim=True)
                xvar = x.var(dim,keepdim=True)
            else:
                xmean = self.running_mean
                xvar = self.running_var
            # now we will apply the normalization formula that we discussed above
            xhat = (x - xmean) / torch.sqrt(xvar+self.eps)
            # gamma and beta are learnable and so it will allow model to move and play around with the distribution
            self.output = xhat*self.gamma + self.beta

            if self.is_training:
                # update the running mean and variance AND we dont want to keep gradients for the following computations in our                          computational graph 
                with torch.no_grad():
                    self.running_mean = (1-self.momentum)*self.running_mean + (self.momentum)*xmean 
                    self.running_var  = (1-self.momentum)*self.running_var  + (self.momentum)*xvar
            return self.output

        def parameters(self):
            return [self.gamma, self.beta]
    return (BatchNorm1d,)


@app.cell
def _(mo):
    mo.md(
        r"""
        #### Code up Tanh
        This should be easy
        [Take a look at Pytorch Tanh Impl](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html)

        $$\text{Tanh}(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}$$
        """
    )
    return


@app.cell
def _(torch):
    class Tanh:
        '''Shapes that this layer works with 
        Input: any number of dimensions
        Output: Same shape as Input'''
        def __call__(self,x):
            self.output = torch.tanh(x)
            return self.output
        def parameters(self):
            # activation layer doesnot have any learnable parameters
            return []
    return (Tanh,)


@app.cell
def _(mo):
    mo.md(
        r"""
        #### Code up the Embedding Layer
        [Take a look at the pytorch impl of Embedding layer](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
        """
    )
    return


@app.cell
def _(torch):
    # scratch pad 
    nembed = 10 
    embed_dim = 3
    # learnable matrix that will map the input to embedding space 
    mapping_matrix = torch.randn((nembed,embed_dim))
    print(mapping_matrix, mapping_matrix.shape)
    # our input indices 
    input_indices = torch.tensor([1,2,3])
    print(f"input indices: {input_indices, input_indices.shape}")
    out_embed = mapping_matrix[0,1] # we dont want this 
    print(f"output indices:\n{out_embed,out_embed.shape}") 
    out_embed = mapping_matrix[[0,1]] # we want something like this 
    print(f"output indices:\n{out_embed,out_embed.shape}")
    out_embed = mapping_matrix[input_indices]
    print(f"output indices:\n{out_embed,out_embed.shape}")
    return embed_dim, input_indices, mapping_matrix, nembed, out_embed


@app.cell
def _(torch):
    class Embedding:
        '''A simple lookup table that stores embeddings of a fixed dictionary and size.
            num_embeddings: size of the vocabulary 
            embedding_dim : size of each embedding vector
        '''
        def __init__(self, num_embeddings:int, embedding_dim:int):
            # this weight matrix is learnable
            self.weight = torch.randn((num_embeddings,embedding_dim))
        def __call__(self, indexes):
            self.output = self.weight[indexes]
            return self.output
        def parameters(self):
            return [self.weight]
    return (Embedding,)


@app.cell
def _(mo):
    mo.md(
        r"""
        #### Code up Flatten Layer
        [Take a look at the torch impl of Flattening](https://pytorch.org/docs/stable/generated/torch.flatten.html)

        Our implementation will be slighlty different. Firstly we expect the input to this layer to have the shape [N,L,C]
        which mean first dim is batch size, second dim is the sequence length (time series) and finally we have the feature.

        As we saw the wavenet architecture, we want to group together the sequences along the time dimension but then it would also mean that the resulting grouped tensor will have the features from the combined sequence. 

        Say we want to make groups of 2 or combine two consecutive sequences into 1?
        """
    )
    return


@app.cell
def _(mo):
    mo.image("groupconsecutive.png")
    return


@app.cell
def _(torch):
    # scratch pad
    def _():
        U = torch.randn((3,6,10))
        time_step_stride = 2

        N,L,C = U.shape
        print(f"U shape before grouping: {N},{L},{C}")
        GroupedU = U.view(N,  L//time_step_stride,  C*time_step_stride)

        N,L,C = GroupedU.shape
        print(f"U shape after grouping: {N},{L},{C}")

        # lets test some edge cases 
        # suppose at the end we only have a sequence with length 1, it is unneccasary to have that dimension 
        batch = torch.randn(10, 1, 5)
        # how to remove ? let see 
        print(batch.unsqueeze(0).shape) # unsqueeze added a new dim at 0th index and shifted all other dim to right 
        print(batch.squeeze(1).shape) # okay this is what we need, we want to remove the 'Singleton dim'

        # Let's see how padding would work if L is not dividible by time step stride 
        # N,L,C = 8,5,20 
        # so 5 is not divisible by 2 directly and we need to perform padding such that is divisible 
        x = torch.randn((8,5,20))
        N,L,C = x.shape
        print(f" x shape is: {x.shape}")
        # the two lines below will give error because the L is not evenly divisble by 2 so we need to pad 
        # y = x.view(N,L//2,C*2)
        # print(f" y shape is: {y.shape}")
        # assuming time step stride is 2
        padding = L % 2 
        print(f" padding: {padding}")


    _()
    return


@app.cell
def _(F):
    class GroupConsecutive:
        """ this layer will help you group together sequences 
        time_step_stride: represents the number of consecutive time steps you want to combine in each group.
        """
        def __init__(self, time_step_stride:int):
            self.time_step_stride = time_step_stride

        def __call__(self,x):
            # lets handle 2d inputs as well, this might happen in some layers where previous operation of group consec squeezed the L dim
            N,L,C = x.shape

            # padding, assumption here is that timestepstride is always 2
            if L%self.time_step_stride !=0:
                x = F.pad(x, (0,0,0,1), "constant", 0)

            num_groups = x.shape[1]//self.time_step_stride
            x=x.view(N, num_groups, C*self.time_step_stride)

            # squeeze is not needed since we are correctly choosing the last time step when computing the logits
            # if (x.shape[1] == 1):
            #     x = x.squeeze(1)

            self.output = x 
            return self.output

        def parameters(self):
            # doesnot have any learnable params 
            return []
    return (GroupConsecutive,)


@app.cell
def _(mo):
    mo.md(
        r"""
        #### let's code the Sequential Container 
        All the layers are added sequentially into this container ⛓️ and are connected in a cascading way.
        [Take a look at how pytorch implements it](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)
        """
    )
    return


@app.cell
def _():
    # scratch pad 
    def _():
        # lets also discuss the diff btw generator exp and list exp 
        list_res = [x for x in range(5)]
        print(list_res)
        print(list_res[2])
        print(len(list_res))
        print('--LIST exp---')
        # we can iterator a list expr however many times 
        for x in list_res:
            print(x)
        print('-----')
        for x in list_res:
            print(x)
        print('--GEN exp---')
        # generator exp
        gen_res = (x for x in range(5))
        print(gen_res) # return a gen obj
        # gen can be iterated only once
        # this gives error print(gen_res[0])
        for x in gen_res:
            print(x)
        print('--Will be empty now since gen is used up now---')
        for x in gen_res:
            print(x)
        print('-----')
        # conclusion: generator expression are useful for large datasets because they generate values on the fly instead of storing them all at once like a list
        class layer:
            def parameters(self):
                return [1] # cant simply return 1, cuz list comprehension expects an iterable like a list or generator that can be iterated over 
        x = layer()
        y = layer() 
        layers=[]
        layers.append(x)
        layers.append(y)
        print([layer.parameters()[0] for layer in layers])
        # print([p for layer in layers for p in layer.parameters()])

    _()
    return


@app.cell
def _():
    class Sequential:
        '''a container to hold the layers and call the forward method on each 
        '''
        def __init__(self, layers):
            self.layers = layers

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
                #print(f"{layer.__class__.__name__}: {x.shape}") # for debugging 
            self.output = x 
            return self.output

        def parameters(self):
            return [p for layer in self.layers for p in layer.parameters()]
    return (Sequential,)


@app.cell
def _(mo):
    mo.md(
        r"""
        #### Let's code up the Network
        1. Decide the embedding dimensionality of the character embedding vectors
        2. Decide the number of neurons in the hidden layer of the MLP
        3. Define the model, choose the order and number of each layer
        4. Begin Training
        5. Perform Train/Validation Loss analysis
        6. Sample from Model
        """
    )
    return


@app.cell
def _(mo):
    embedding_dimension = mo.ui.slider(start=10, stop=50, step=2)
    embedding_dimension
    return (embedding_dimension,)


@app.cell
def _(mo):
    hidden_layer_size = mo.ui.slider(start=100, stop=200, step=2)
    hidden_layer_size
    return (hidden_layer_size,)


@app.cell
def _(
    BatchNorm1d,
    Embedding,
    GroupConsecutive,
    Linear,
    Sequential,
    Tanh,
    embedding_dimension,
    hidden_layer_size,
    torch,
    vocabulary_size,
):
    model = Sequential([
            Embedding(vocabulary_size,embedding_dimension.value),

            GroupConsecutive(2), 
            Linear(embedding_dimension.value*2,hidden_layer_size.value,bias=False),
            BatchNorm1d(hidden_layer_size.value),
            Tanh(),

            GroupConsecutive(2),
            Linear(hidden_layer_size.value  *2,hidden_layer_size.value,bias=False),
            BatchNorm1d(hidden_layer_size.value), 
            Tanh(),

            GroupConsecutive(2),
            Linear(hidden_layer_size.value  *2,hidden_layer_size.value,bias=False),
            BatchNorm1d(hidden_layer_size.value), 
            Tanh(),

            Linear(hidden_layer_size.value,vocabulary_size),
        ])

    with torch.no_grad():
        model.layers[-1].weight *= 0.1
    return (model,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ##### you might wonder why did we multiply the weight of the last layer with 0.1 ? 

        At the beginning of training we dont want any particular character to have a high precedence over other characters so all the logits should have roughly same value hence roughly a uniform distribution at the output layer
        """
    )
    return


@app.cell
def _(model):
    # Count the total parameters of the model 
    parameters = model.parameters()
    params = [p.nelement() for p in parameters]
    print(f"MODEL PARAMS: {sum(params)}")
    # prepare for training 
    for p in parameters:
        p.requires_grad = True
    return p, parameters, params


@app.cell
def _(mo):
    mo.md(r"""#### Lets define some training parameters that are tuneable""")
    return


@app.cell
def _(mo):
    max_iterations = mo.ui.slider(start=10000, stop=200000, step=1)
    max_iterations
    return (max_iterations,)


@app.cell
def _(mo):
    batch_size = mo.ui.slider(start=8, stop=64, step=8)
    batch_size
    return (batch_size,)


@app.cell
def _(mo):
    mo.md(r"""#### Training Loop""")
    return


@app.cell
def _(torch):
    # scratch pad 
    # (1) is just an integer in paranthesis 
    # (1,) is a tuple containing single element
    def _():
        a = torch.randint(0,200,(32,))
        # choose a random batch from our data set
        print(a)
        # remember our X train was torch.Size([205907, 3]) if context is 3, lets select batch size as 32
        index = torch.randint(0, 205907, (32,))
        print(index)
        # lets see a dummy example 
        x = torch.randn(5,3) # xtrain
        print(x) 
        index = torch.randint(0,5,(2,)) # randomly choosen batch of 2 examples 
        print(index)
        print(x[index])
        x = torch.randn((8,2,53))
        print(x.shape)
        b = x[:,-1,:] # use only the last time step
        print(b.shape)
    _()
    return


@app.cell
def _(F, Xtrain, Ytrain, batch_size, max_iterations, model, parameters, torch):
    iteration_loss = []
    for i in range(max_iterations.value):
        index = torch.randint(0, Xtrain.shape[0],(batch_size.value,))
        Xbatch,Ybatch = Xtrain[index], Ytrain[index]

        # forward pass 
        logits = model(Xbatch)
        logits = logits[:, -1, :]
        loss = F.cross_entropy(logits, Ybatch)

        # backward pass: keep in mind that when performing backprop clear up all the previously computed gradients 
        # You should not accumulate gradients 
        for param in parameters:
            param.grad = None
        loss.backward()

        lr = 0.1 if i < 50000 else 0.01
        for param in parameters:
            param.data += -lr* param.grad

        if i%1000 == 0:
            print(f'{i:7d}/{max_iterations.value:7d}: {loss.item():.4f}')
        iteration_loss.append(loss.log10().item())
    return Xbatch, Ybatch, i, index, iteration_loss, logits, loss, lr, param


@app.cell
def _(iteration_loss, torch):
    # scratch pad 
    import matplotlib.pyplot as plt
    def _():
        plt.figure(figsize=(10, 5))  
        plt.plot(torch.tensor(iteration_loss).view(-1,5000).mean(1))  
    
        plt.title('Training Loss over Iterations')   
        return plt.gcf()
    _()
    return (plt,)


@app.cell
def _(mo):
    mo.md(
        r"""
        #### Evaluate 
        In evaluation stage, we must first set the training equals to 'False' for our layers especially for the Batch Norm layer since that has different behavior in evaluation and training
        """
    )
    return


@app.cell
def _():
    # scratch pad 
    def _():
        split = ['train']
        dict = {'train':{'a','b'}}
        print(dict)
        x,y = dict['train']
        print(x,y)
    _()
    return


@app.cell
def _(F, Xtest, Xtrain, Xval, Ytest, Ytrain, Yval, model, torch):
    for layer in model.layers:
        layer.is_training = False

    @torch.no_grad()
    def evaluation(split):
        x,y = {
            'train':{Xtrain,Ytrain},
            'val':{Xval,Yval},
            'test':{Xtest,Ytest}
        }[split]
        logits = model(x)
        logits = logits[:,-1,:]
        loss = F.cross_entropy(logits,y)
        print(split, loss.item())

    evaluation('train')
    evaluation('val')
    evaluation('test')
    return evaluation, layer


@app.cell
def _(mo):
    mo.md(r"""#### Sample from the model""")
    return


@app.cell
def _(torch):
    #scratch pad 
    def _():
        torch.set_printoptions(  
        precision=4,       
        sci_mode=False,    
        linewidth=120,     
        edgeitems=3)
        random = torch.abs(torch.randn((1,5)))
        print(random, random.shape)
        sum_across_cols = torch.sum(random,dim=1,keepdim=True)
        print(sum_across_cols, sum_across_cols.shape)
        prob = random/sum_across_cols
        print(prob,prob.shape)
        sample_index_based_on_prob = torch.multinomial(prob, num_samples=1).item()
        print(sample_index_based_on_prob)
    _()

    return


@app.cell
def _(F, context_slider, integer_to_string, model, torch):
    for _ in range(20):
        out = []
        context = [0] * context_slider.value
        while True:
            lgits = model(torch.tensor([context]))
            lgits = lgits[:,-1,:]
            probabilities = F.softmax(lgits, dim=1)
            ix = torch.multinomial(probabilities, num_samples=1).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix==0:
                break
        print(''.join(integer_to_string[i] for i in out))
        
    return context, ix, lgits, out, probabilities


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
