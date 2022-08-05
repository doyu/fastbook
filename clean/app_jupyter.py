#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#hide
get_ipython().system(' [ -e /content ] && pip install -Uqq fastbook')
import fastbook
fastbook.setup_book()


# In[ ]:


from fastbook import *


# # Appendix: Jupyter Notebook 101

# ## Introduction

# In[ ]:


1+1


# ## Writing

# In[ ]:


3/2


# ## Modes

# ## Other Important Considerations

# ## Markdown Formatting
# 

# ### Italics, Bold, Strikethrough, Inline, Blockquotes and Links

# ### Headings

# ### Lists

# ## Code Capabilities

# In[ ]:


# Import necessary libraries
from fastai.vision.all import * 
import matplotlib.pyplot as plt


# In[ ]:


from PIL import Image


# In[ ]:


a = 1
b = a + 1
c = b + a + 1
d = c + b + a + 1
a, b, c ,d


# In[ ]:


plt.plot([a,b,c,d])
plt.show()


# In[ ]:


Image.open(image_cat())


# ## Running the App Locally

# ## Creating a Notebook

# ## Shortcuts and Tricks

# ### Command Mode Shortcuts

# ### Cell Tricks

# ### Line Magics

# In[ ]:


get_ipython().run_line_magic('timeit', '[i+1 for i in range(1000)]')

