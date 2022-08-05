#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#hide
get_ipython().system(' [ -e /content ] && pip install -Uqq fastbook')
import fastbook
fastbook.setup_book()


# In[ ]:


#hide
from fastai.vision.all import *
from fastbook import *

matplotlib.rc('image', cmap='Greys')


# # Convolutional Neural Networks

# ## The Magic of Convolutions

# In[ ]:


top_edge = tensor([[-1,-1,-1],
                   [ 0, 0, 0],
                   [ 1, 1, 1]]).float()


# In[ ]:


path = untar_data(URLs.MNIST_SAMPLE)


# In[ ]:


#hide
Path.BASE_PATH = path


# In[ ]:


im3 = Image.open(path/'train'/'3'/'12.png')
show_image(im3);


# In[ ]:


im3_t = tensor(im3)
im3_t[0:3,0:3] * top_edge


# In[ ]:


(im3_t[0:3,0:3] * top_edge).sum()


# In[ ]:


df = pd.DataFrame(im3_t[:10,:20])
df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')


# In[ ]:


(im3_t[4:7,6:9] * top_edge).sum()


# In[ ]:


(im3_t[7:10,17:20] * top_edge).sum()


# In[ ]:


def apply_kernel(row, col, kernel):
    return (im3_t[row-1:row+2,col-1:col+2] * kernel).sum()


# In[ ]:


apply_kernel(5,7,top_edge)


# ### Mapping a Convolution Kernel

# In[ ]:


[[(i,j) for j in range(1,5)] for i in range(1,5)]


# In[ ]:


rng = range(1,27)
top_edge3 = tensor([[apply_kernel(i,j,top_edge) for j in rng] for i in rng])

show_image(top_edge3);


# In[ ]:


left_edge = tensor([[-1,1,0],
                    [-1,1,0],
                    [-1,1,0]]).float()

left_edge3 = tensor([[apply_kernel(i,j,left_edge) for j in rng] for i in rng])

show_image(left_edge3);


# ### Convolutions in PyTorch

# In[ ]:


diag1_edge = tensor([[ 0,-1, 1],
                     [-1, 1, 0],
                     [ 1, 0, 0]]).float()
diag2_edge = tensor([[ 1,-1, 0],
                     [ 0, 1,-1],
                     [ 0, 0, 1]]).float()

edge_kernels = torch.stack([left_edge, top_edge, diag1_edge, diag2_edge])
edge_kernels.shape


# In[ ]:


mnist = DataBlock((ImageBlock(cls=PILImageBW), CategoryBlock), 
                  get_items=get_image_files, 
                  splitter=GrandparentSplitter(),
                  get_y=parent_label)

dls = mnist.dataloaders(path)
xb,yb = first(dls.valid)
xb.shape


# In[ ]:


xb,yb = to_cpu(xb),to_cpu(yb)


# In[ ]:


edge_kernels.shape,edge_kernels.unsqueeze(1).shape


# In[ ]:


edge_kernels = edge_kernels.unsqueeze(1)


# In[ ]:


batch_features = F.conv2d(xb, edge_kernels)
batch_features.shape


# In[ ]:


show_image(batch_features[0,0]);


# ### Strides and Padding

# ### Understanding the Convolution Equations

# ## Our First Convolutional Neural Network

# ### Creating the CNN

# In[ ]:


simple_net = nn.Sequential(
    nn.Linear(28*28,30),
    nn.ReLU(),
    nn.Linear(30,1)
)


# In[ ]:


simple_net


# In[ ]:


broken_cnn = sequential(
    nn.Conv2d(1,30, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(30,1, kernel_size=3, padding=1)
)


# In[ ]:


broken_cnn(xb).shape


# In[ ]:


def conv(ni, nf, ks=3, act=True):
    res = nn.Conv2d(ni, nf, stride=2, kernel_size=ks, padding=ks//2)
    if act: res = nn.Sequential(res, nn.ReLU())
    return res


# In[ ]:


simple_cnn = sequential(
    conv(1 ,4),            #14x14
    conv(4 ,8),            #7x7
    conv(8 ,16),           #4x4
    conv(16,32),           #2x2
    conv(32,2, act=False), #1x1
    Flatten(),
)


# In[ ]:


simple_cnn(xb).shape


# In[ ]:


learn = Learner(dls, simple_cnn, loss_func=F.cross_entropy, metrics=accuracy)


# In[ ]:


learn.summary()


# In[ ]:


learn.fit_one_cycle(2, 0.01)


# ### Understanding Convolution Arithmetic

# In[ ]:


m = learn.model[0]
m


# In[ ]:


m[0].weight.shape


# In[ ]:


m[0].bias.shape


# ### Receptive Fields

# ### A Note About Twitter

# ## Color Images

# In[ ]:


im = image2tensor(Image.open(image_bear()))
im.shape


# In[ ]:


show_image(im);


# In[ ]:


_,axs = subplots(1,3)
for bear,ax,color in zip(im,axs,('Reds','Greens','Blues')):
    show_image(255-bear, ax=ax, cmap=color)


# ## Improving Training Stability

# In[ ]:


path = untar_data(URLs.MNIST)


# In[ ]:


#hide
Path.BASE_PATH = path


# In[ ]:


path.ls()


# In[ ]:


def get_dls(bs=64):
    return DataBlock(
        blocks=(ImageBlock(cls=PILImageBW), CategoryBlock), 
        get_items=get_image_files, 
        splitter=GrandparentSplitter('training','testing'),
        get_y=parent_label,
        batch_tfms=Normalize()
    ).dataloaders(path, bs=bs)

dls = get_dls()


# In[ ]:


dls.show_batch(max_n=9, figsize=(4,4))


# ### A Simple Baseline

# In[ ]:


def conv(ni, nf, ks=3, act=True):
    res = nn.Conv2d(ni, nf, stride=2, kernel_size=ks, padding=ks//2)
    if act: res = nn.Sequential(res, nn.ReLU())
    return res


# In[ ]:


def simple_cnn():
    return sequential(
        conv(1 ,8, ks=5),        #14x14
        conv(8 ,16),             #7x7
        conv(16,32),             #4x4
        conv(32,64),             #2x2
        conv(64,10, act=False),  #1x1
        Flatten(),
    )


# In[ ]:


from fastai.callback.hook import *


# In[ ]:


def fit(epochs=1):
    learn = Learner(dls, simple_cnn(), loss_func=F.cross_entropy,
                    metrics=accuracy, cbs=ActivationStats(with_hist=True))
    learn.fit(epochs, 0.06)
    return learn


# In[ ]:


learn = fit()


# In[ ]:


learn.activation_stats.plot_layer_stats(0)


# In[ ]:


learn.activation_stats.plot_layer_stats(-2)


# ### Increase Batch Size

# In[ ]:


dls = get_dls(512)


# In[ ]:


learn = fit()


# In[ ]:


learn.activation_stats.plot_layer_stats(-2)


# ### 1cycle Training

# In[ ]:


def fit(epochs=1, lr=0.06):
    learn = Learner(dls, simple_cnn(), loss_func=F.cross_entropy,
                    metrics=accuracy, cbs=ActivationStats(with_hist=True))
    learn.fit_one_cycle(epochs, lr)
    return learn


# In[ ]:


learn = fit()


# In[ ]:


learn.recorder.plot_sched()


# In[ ]:


learn.activation_stats.plot_layer_stats(-2)


# In[ ]:


learn.activation_stats.color_dim(-2)


# In[ ]:


learn.activation_stats.color_dim(-2)


# ### Batch Normalization

# In[ ]:


def conv(ni, nf, ks=3, act=True):
    layers = [nn.Conv2d(ni, nf, stride=2, kernel_size=ks, padding=ks//2)]
    if act: layers.append(nn.ReLU())
    layers.append(nn.BatchNorm2d(nf))
    return nn.Sequential(*layers)


# In[ ]:


learn = fit()


# In[ ]:


learn.activation_stats.color_dim(-4)


# In[ ]:


learn = fit(5, lr=0.1)


# ## Conclusions

# ## Questionnaire

# 1. What is a "feature"?
# 1. Write out the convolutional kernel matrix for a top edge detector.
# 1. Write out the mathematical operation applied by a 3×3 kernel to a single pixel in an image.
# 1. What is the value of a convolutional kernel apply to a 3×3 matrix of zeros?
# 1. What is "padding"?
# 1. What is "stride"?
# 1. Create a nested list comprehension to complete any task that you choose.
# 1. What are the shapes of the `input` and `weight` parameters to PyTorch's 2D convolution?
# 1. What is a "channel"?
# 1. What is the relationship between a convolution and a matrix multiplication?
# 1. What is a "convolutional neural network"?
# 1. What is the benefit of refactoring parts of your neural network definition?
# 1. What is `Flatten`? Where does it need to be included in the MNIST CNN? Why?
# 1. What does "NCHW" mean?
# 1. Why does the third layer of the MNIST CNN have `7*7*(1168-16)` multiplications?
# 1. What is a "receptive field"?
# 1. What is the size of the receptive field of an activation after two stride 2 convolutions? Why?
# 1. Run *conv-example.xlsx* yourself and experiment with *trace precedents*.
# 1. Have a look at Jeremy or Sylvain's list of recent Twitter "like"s, and see if you find any interesting resources or ideas there.
# 1. How is a color image represented as a tensor?
# 1. How does a convolution work with a color input?
# 1. What method can we use to see that data in `DataLoaders`?
# 1. Why do we double the number of filters after each stride-2 conv?
# 1. Why do we use a larger kernel in the first conv with MNIST (with `simple_cnn`)?
# 1. What information does `ActivationStats` save for each layer?
# 1. How can we access a learner's callback after training?
# 1. What are the three statistics plotted by `plot_layer_stats`? What does the x-axis represent?
# 1. Why are activations near zero problematic?
# 1. What are the upsides and downsides of training with a larger batch size?
# 1. Why should we avoid using a high learning rate at the start of training?
# 1. What is 1cycle training?
# 1. What are the benefits of training with a high learning rate?
# 1. Why do we want to use a low learning rate at the end of training?
# 1. What is "cyclical momentum"?
# 1. What callback tracks hyperparameter values during training (along with other information)?
# 1. What does one column of pixels in the `color_dim` plot represent?
# 1. What does "bad training" look like in `color_dim`? Why?
# 1. What trainable parameters does a batch normalization layer contain?
# 1. What statistics are used to normalize in batch normalization during training? How about during validation?
# 1. Why do models with batch normalization layers generalize better?

# ### Further Research

# 1. What features other than edge detectors have been used in computer vision (especially before deep learning became popular)?
# 1. There are other normalization layers available in PyTorch. Try them out and see what works best. Learn about why other normalization layers have been developed, and how they differ from batch normalization.
# 1. Try moving the activation function after the batch normalization layer in `conv`. Does it make a difference? See what you can find out about what order is recommended, and why.

# In[ ]:




