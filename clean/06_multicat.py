#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#hide
get_ipython().system(' [ -e /content ] && pip install -Uqq fastbook')
import fastbook
fastbook.setup_book()


# In[ ]:


#hide
from fastbook import *


# # Other Computer Vision Problems

# ## Multi-Label Classification

# ### The Data

# In[ ]:


from fastai.vision.all import *
path = untar_data(URLs.PASCAL_2007)


# In[ ]:


df = pd.read_csv(path/'train.csv')
df.head()


# ### Sidebar: Pandas and DataFrames

# In[ ]:


df.iloc[:,0]


# In[ ]:


df.iloc[0,:]
# Trailing :s are always optional (in numpy, pytorch, pandas, etc.),
#   so this is equivalent:
df.iloc[0]


# In[ ]:


df['fname']


# In[ ]:


tmp_df = pd.DataFrame({'a':[1,2], 'b':[3,4]})
tmp_df


# In[ ]:


tmp_df['c'] = tmp_df['a']+tmp_df['b']
tmp_df


# ### End sidebar

# ### Constructing a DataBlock

# In[ ]:


dblock = DataBlock()


# In[ ]:


dsets = dblock.datasets(df)


# In[ ]:


len(dsets.train),len(dsets.valid)


# In[ ]:


x,y = dsets.train[0]
x,y


# In[ ]:


x['fname']


# In[ ]:


dblock = DataBlock(get_x = lambda r: r['fname'], get_y = lambda r: r['labels'])
dsets = dblock.datasets(df)
dsets.train[0]


# In[ ]:


def get_x(r): return r['fname']
def get_y(r): return r['labels']
dblock = DataBlock(get_x = get_x, get_y = get_y)
dsets = dblock.datasets(df)
dsets.train[0]


# In[ ]:


def get_x(r): return path/'train'/r['fname']
def get_y(r): return r['labels'].split(' ')
dblock = DataBlock(get_x = get_x, get_y = get_y)
dsets = dblock.datasets(df)
dsets.train[0]


# In[ ]:


dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   get_x = get_x, get_y = get_y)
dsets = dblock.datasets(df)
dsets.train[0]


# In[ ]:


idxs = torch.where(dsets.train[0][1]==1.)[0]
dsets.train.vocab[idxs]


# In[ ]:


def splitter(df):
    train = df.index[~df['is_valid']].tolist()
    valid = df.index[df['is_valid']].tolist()
    return train,valid

dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=splitter,
                   get_x=get_x, 
                   get_y=get_y)

dsets = dblock.datasets(df)
dsets.train[0]


# In[ ]:


dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=splitter,
                   get_x=get_x, 
                   get_y=get_y,
                   item_tfms = RandomResizedCrop(128, min_scale=0.35))
dls = dblock.dataloaders(df)


# In[ ]:


dls.show_batch(nrows=1, ncols=3)


# ### Binary Cross-Entropy

# In[ ]:


learn = vision_learner(dls, resnet18)


# In[ ]:


x,y = to_cpu(dls.train.one_batch())
activs = learn.model(x)
activs.shape


# In[ ]:


activs[0]


# In[ ]:


def binary_cross_entropy(inputs, targets):
    inputs = inputs.sigmoid()
    return -torch.where(targets==1, inputs, 1-inputs).log().mean()


# In[ ]:


loss_func = nn.BCEWithLogitsLoss()
loss = loss_func(activs, y)
loss


# In[ ]:


def say_hello(name, say_what="Hello"): return f"{say_what} {name}."
say_hello('Jeremy'),say_hello('Jeremy', 'Ahoy!')


# In[ ]:


f = partial(say_hello, say_what="Bonjour")
f("Jeremy"),f("Sylvain")


# In[ ]:


learn = vision_learner(dls, resnet50, metrics=partial(accuracy_multi, thresh=0.2))
learn.fine_tune(3, base_lr=3e-3, freeze_epochs=4)


# In[ ]:


learn.metrics = partial(accuracy_multi, thresh=0.1)
learn.validate()


# In[ ]:


learn.metrics = partial(accuracy_multi, thresh=0.99)
learn.validate()


# In[ ]:


preds,targs = learn.get_preds()


# In[ ]:


accuracy_multi(preds, targs, thresh=0.9, sigmoid=False)


# In[ ]:


xs = torch.linspace(0.05,0.95,29)
accs = [accuracy_multi(preds, targs, thresh=i, sigmoid=False) for i in xs]
plt.plot(xs,accs);


# ## Regression

# ### Assemble the Data

# In[ ]:


path = untar_data(URLs.BIWI_HEAD_POSE)


# In[ ]:


#hide
Path.BASE_PATH = path


# In[ ]:


path.ls().sorted()


# In[ ]:


(path/'01').ls().sorted()


# In[ ]:


img_files = get_image_files(path)
def img2pose(x): return Path(f'{str(x)[:-7]}pose.txt')
img2pose(img_files[0])


# In[ ]:


im = PILImage.create(img_files[0])
im.shape


# In[ ]:


im.to_thumb(160)


# In[ ]:


cal = np.genfromtxt(path/'01'/'rgb.cal', skip_footer=6)
def get_ctr(f):
    ctr = np.genfromtxt(img2pose(f), skip_header=3)
    c1 = ctr[0] * cal[0][0]/ctr[2] + cal[0][2]
    c2 = ctr[1] * cal[1][1]/ctr[2] + cal[1][2]
    return tensor([c1,c2])


# In[ ]:


get_ctr(img_files[0])


# In[ ]:


biwi = DataBlock(
    blocks=(ImageBlock, PointBlock),
    get_items=get_image_files,
    get_y=get_ctr,
    splitter=FuncSplitter(lambda o: o.parent.name=='13'),
    batch_tfms=aug_transforms(size=(240,320)), 
)


# In[ ]:


dls = biwi.dataloaders(path)
dls.show_batch(max_n=9, figsize=(8,6))


# In[ ]:


xb,yb = dls.one_batch()
xb.shape,yb.shape


# In[ ]:


yb[0]


# ### Training a Model

# In[ ]:


learn = vision_learner(dls, resnet18, y_range=(-1,1))


# In[ ]:


def sigmoid_range(x, lo, hi): return torch.sigmoid(x) * (hi-lo) + lo


# In[ ]:


plot_function(partial(sigmoid_range,lo=-1,hi=1), min=-4, max=4)


# In[ ]:


dls.loss_func


# In[ ]:


learn.lr_find()


# In[ ]:


lr = 1e-2
learn.fine_tune(3, lr)


# In[ ]:


math.sqrt(0.0001)


# In[ ]:


learn.show_results(ds_idx=1, nrows=3, figsize=(6,8))


# ## Conclusion

# ## Questionnaire

# 1. How could multi-label classification improve the usability of the bear classifier?
# 1. How do we encode the dependent variable in a multi-label classification problem?
# 1. How do you access the rows and columns of a DataFrame as if it was a matrix?
# 1. How do you get a column by name from a DataFrame?
# 1. What is the difference between a `Dataset` and `DataLoader`?
# 1. What does a `Datasets` object normally contain?
# 1. What does a `DataLoaders` object normally contain?
# 1. What does `lambda` do in Python?
# 1. What are the methods to customize how the independent and dependent variables are created with the data block API?
# 1. Why is softmax not an appropriate output activation function when using a one hot encoded target?
# 1. Why is `nll_loss` not an appropriate loss function when using a one-hot-encoded target?
# 1. What is the difference between `nn.BCELoss` and `nn.BCEWithLogitsLoss`?
# 1. Why can't we use regular accuracy in a multi-label problem?
# 1. When is it okay to tune a hyperparameter on the validation set?
# 1. How is `y_range` implemented in fastai? (See if you can implement it yourself and test it without peeking!)
# 1. What is a regression problem? What loss function should you use for such a problem?
# 1. What do you need to do to make sure the fastai library applies the same data augmentation to your input images and your target point coordinates?

# ### Further Research

# 1. Read a tutorial about Pandas DataFrames and experiment with a few methods that look interesting to you. See the book's website for recommended tutorials.
# 1. Retrain the bear classifier using multi-label classification. See if you can make it work effectively with images that don't contain any bears, including showing that information in the web application. Try an image with two different kinds of bears. Check whether the accuracy on the single-label dataset is impacted using multi-label classification.

# In[ ]:




