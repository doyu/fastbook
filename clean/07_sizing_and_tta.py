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


# # Training a State-of-the-Art Model

# ## Imagenette

# In[ ]:


from fastai.vision.all import *
path = untar_data(URLs.IMAGENETTE)


# In[ ]:


dblock = DataBlock(blocks=(ImageBlock(), CategoryBlock()),
                   get_items=get_image_files,
                   get_y=parent_label,
                   item_tfms=Resize(460),
                   batch_tfms=aug_transforms(size=224, min_scale=0.75))
dls = dblock.dataloaders(path, bs=64)


# In[ ]:


model = xresnet50(n_out=dls.c)
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
learn.fit_one_cycle(5, 3e-3)


# ## Normalization

# In[ ]:


x,y = dls.one_batch()
x.mean(dim=[0,2,3]),x.std(dim=[0,2,3])


# In[ ]:


def get_dls(bs, size):
    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   get_items=get_image_files,
                   get_y=parent_label,
                   item_tfms=Resize(460),
                   batch_tfms=[*aug_transforms(size=size, min_scale=0.75),
                               Normalize.from_stats(*imagenet_stats)])
    return dblock.dataloaders(path, bs=bs)


# In[ ]:


dls = get_dls(64, 224)


# In[ ]:


x,y = dls.one_batch()
x.mean(dim=[0,2,3]),x.std(dim=[0,2,3])


# In[ ]:


model = xresnet50(n_out=dls.c)
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
learn.fit_one_cycle(5, 3e-3)


# ## Progressive Resizing

# In[ ]:


dls = get_dls(128, 128)
learn = Learner(dls, xresnet50(n_out=dls.c), loss_func=CrossEntropyLossFlat(), 
                metrics=accuracy)
learn.fit_one_cycle(4, 3e-3)


# In[ ]:


learn.dls = get_dls(64, 224)
learn.fine_tune(5, 1e-3)


# ## Test Time Augmentation

# In[ ]:


preds,targs = learn.tta()
accuracy(preds, targs).item()


# ## Mixup

# ### Sidebar: Papers and Math

# ### End sidebar

# In[ ]:


church = PILImage.create(get_image_files_sorted(path/'train'/'n03028079')[0])
gas = PILImage.create(get_image_files_sorted(path/'train'/'n03425413')[0])
church = church.resize((256,256))
gas = gas.resize((256,256))
tchurch = tensor(church).float() / 255.
tgas = tensor(gas).float() / 255.

_,axs = plt.subplots(1, 3, figsize=(12,4))
show_image(tchurch, ax=axs[0]);
show_image(tgas, ax=axs[1]);
show_image((0.3*tchurch + 0.7*tgas), ax=axs[2]);


# ## Label Smoothing

# ### Sidebar: Label Smoothing, the Paper

# ### End sidebar

# ## Conclusion

# ## Questionnaire

# 1. What is the difference between ImageNet and Imagenette? When is it better to experiment on one versus the other?
# 1. What is normalization?
# 1. Why didn't we have to care about normalization when using a pretrained model?
# 1. What is progressive resizing?
# 1. Implement progressive resizing in your own project. Did it help?
# 1. What is test time augmentation? How do you use it in fastai?
# 1. Is using TTA at inference slower or faster than regular inference? Why?
# 1. What is Mixup? How do you use it in fastai?
# 1. Why does Mixup prevent the model from being too confident?
# 1. Why does training with Mixup for five epochs end up worse than training without Mixup?
# 1. What is the idea behind label smoothing?
# 1. What problems in your data can label smoothing help with?
# 1. When using label smoothing with five categories, what is the target associated with the index 1?
# 1. What is the first step to take when you want to prototype quick experiments on a new dataset?

# ### Further Research
# 
# 1. Use the fastai documentation to build a function that crops an image to a square in each of the four corners, then implement a TTA method that averages the predictions on a center crop and those four crops. Did it help? Is it better than the TTA method of fastai?
# 1. Find the Mixup paper on arXiv and read it. Pick one or two more recent articles introducing variants of Mixup and read them, then try to implement them on your problem.
# 1. Find the script training Imagenette using Mixup and use it as an example to build a script for a long training on your own project. Execute it and see if it helps.
# 1. Read the sidebar "Label Smoothing, the Paper", look at the relevant section of the original paper and see if you can follow it. Don't be afraid to ask for help!

# In[ ]:




