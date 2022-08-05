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


# # Collaborative Filtering Deep Dive

# ## A First Look at the Data

# In[ ]:


from fastai.collab import *
from fastai.tabular.all import *
path = untar_data(URLs.ML_100k)


# In[ ]:


ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None,
                      names=['user','movie','rating','timestamp'])
ratings.head()


# In[ ]:


last_skywalker = np.array([0.98,0.9,-0.9])


# In[ ]:


user1 = np.array([0.9,0.8,-0.6])


# In[ ]:


(user1*last_skywalker).sum()


# In[ ]:


casablanca = np.array([-0.99,-0.3,0.8])


# In[ ]:


(user1*casablanca).sum()


# ## Learning the Latent Factors

# ## Creating the DataLoaders

# In[ ]:


movies = pd.read_csv(path/'u.item',  delimiter='|', encoding='latin-1',
                     usecols=(0,1), names=('movie','title'), header=None)
movies.head()


# In[ ]:


ratings = ratings.merge(movies)
ratings.head()


# In[ ]:


dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)
dls.show_batch()


# In[ ]:


dls.classes


# In[ ]:


n_users  = len(dls.classes['user'])
n_movies = len(dls.classes['title'])
n_factors = 5

user_factors = torch.randn(n_users, n_factors)
movie_factors = torch.randn(n_movies, n_factors)


# In[ ]:


one_hot_3 = one_hot(3, n_users).float()


# In[ ]:


user_factors.t() @ one_hot_3


# In[ ]:


user_factors[3]


# ## Collaborative Filtering from Scratch

# In[ ]:


class Example:
    def __init__(self, a): self.a = a
    def say(self,x): return f'Hello {self.a}, {x}.'


# In[ ]:


ex = Example('Sylvain')
ex.say('nice to meet you')


# In[ ]:


class DotProduct(Module):
    def __init__(self, n_users, n_movies, n_factors):
        self.user_factors = Embedding(n_users, n_factors)
        self.movie_factors = Embedding(n_movies, n_factors)
        
    def forward(self, x):
        users = self.user_factors(x[:,0])
        movies = self.movie_factors(x[:,1])
        return (users * movies).sum(dim=1)


# In[ ]:


x,y = dls.one_batch()
x.shape


# In[ ]:


model = DotProduct(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())


# In[ ]:


learn.fit_one_cycle(5, 5e-3)


# In[ ]:


class DotProduct(Module):
    def __init__(self, n_users, n_movies, n_factors, y_range=(0,5.5)):
        self.user_factors = Embedding(n_users, n_factors)
        self.movie_factors = Embedding(n_movies, n_factors)
        self.y_range = y_range
        
    def forward(self, x):
        users = self.user_factors(x[:,0])
        movies = self.movie_factors(x[:,1])
        return sigmoid_range((users * movies).sum(dim=1), *self.y_range)


# In[ ]:


model = DotProduct(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3)


# In[ ]:


class DotProductBias(Module):
    def __init__(self, n_users, n_movies, n_factors, y_range=(0,5.5)):
        self.user_factors = Embedding(n_users, n_factors)
        self.user_bias = Embedding(n_users, 1)
        self.movie_factors = Embedding(n_movies, n_factors)
        self.movie_bias = Embedding(n_movies, 1)
        self.y_range = y_range
        
    def forward(self, x):
        users = self.user_factors(x[:,0])
        movies = self.movie_factors(x[:,1])
        res = (users * movies).sum(dim=1, keepdim=True)
        res += self.user_bias(x[:,0]) + self.movie_bias(x[:,1])
        return sigmoid_range(res, *self.y_range)


# In[ ]:


model = DotProductBias(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3)


# ### Weight Decay

# In[ ]:


x = np.linspace(-2,2,100)
a_s = [1,2,5,10,50] 
ys = [a * x**2 for a in a_s]
_,ax = plt.subplots(figsize=(8,6))
for a,y in zip(a_s,ys): ax.plot(x,y, label=f'a={a}')
ax.set_ylim([0,5])
ax.legend();


# In[ ]:


model = DotProductBias(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3, wd=0.1)


# ### Creating Our Own Embedding Module

# In[ ]:


class T(Module):
    def __init__(self): self.a = torch.ones(3)

L(T().parameters())


# In[ ]:


class T(Module):
    def __init__(self): self.a = nn.Parameter(torch.ones(3))

L(T().parameters())


# In[ ]:


class T(Module):
    def __init__(self): self.a = nn.Linear(1, 3, bias=False)

t = T()
L(t.parameters())


# In[ ]:


type(t.a.weight)


# In[ ]:


def create_params(size):
    return nn.Parameter(torch.zeros(*size).normal_(0, 0.01))


# In[ ]:


class DotProductBias(Module):
    def __init__(self, n_users, n_movies, n_factors, y_range=(0,5.5)):
        self.user_factors = create_params([n_users, n_factors])
        self.user_bias = create_params([n_users])
        self.movie_factors = create_params([n_movies, n_factors])
        self.movie_bias = create_params([n_movies])
        self.y_range = y_range
        
    def forward(self, x):
        users = self.user_factors[x[:,0]]
        movies = self.movie_factors[x[:,1]]
        res = (users*movies).sum(dim=1)
        res += self.user_bias[x[:,0]] + self.movie_bias[x[:,1]]
        return sigmoid_range(res, *self.y_range)


# In[ ]:


model = DotProductBias(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3, wd=0.1)


# ## Interpreting Embeddings and Biases

# In[ ]:


movie_bias = learn.model.movie_bias.squeeze()
idxs = movie_bias.argsort()[:5]
[dls.classes['title'][i] for i in idxs]


# In[ ]:


idxs = movie_bias.argsort(descending=True)[:5]
[dls.classes['title'][i] for i in idxs]


# In[ ]:


g = ratings.groupby('title')['rating'].count()
top_movies = g.sort_values(ascending=False).index.values[:1000]
top_idxs = tensor([learn.dls.classes['title'].o2i[m] for m in top_movies])
movie_w = learn.model.movie_factors[top_idxs].cpu().detach()
movie_pca = movie_w.pca(3)
fac0,fac1,fac2 = movie_pca.t()
idxs = list(range(50))
X = fac0[idxs]
Y = fac2[idxs]
plt.figure(figsize=(12,12))
plt.scatter(X, Y)
for i, x, y in zip(top_movies[idxs], X, Y):
    plt.text(x,y,i, color=np.random.rand(3)*0.7, fontsize=11)
plt.show()


# ### Using fastai.collab

# In[ ]:


learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5))


# In[ ]:


learn.fit_one_cycle(5, 5e-3, wd=0.1)


# In[ ]:


learn.model


# In[ ]:


movie_bias = learn.model.i_bias.weight.squeeze()
idxs = movie_bias.argsort(descending=True)[:5]
[dls.classes['title'][i] for i in idxs]


# ### Embedding Distance

# In[ ]:


movie_factors = learn.model.i_weight.weight
idx = dls.classes['title'].o2i['Silence of the Lambs, The (1991)']
distances = nn.CosineSimilarity(dim=1)(movie_factors, movie_factors[idx][None])
idx = distances.argsort(descending=True)[1]
dls.classes['title'][idx]


# ## Bootstrapping a Collaborative Filtering Model

# ## Deep Learning for Collaborative Filtering

# In[ ]:


embs = get_emb_sz(dls)
embs


# In[ ]:


class CollabNN(Module):
    def __init__(self, user_sz, item_sz, y_range=(0,5.5), n_act=100):
        self.user_factors = Embedding(*user_sz)
        self.item_factors = Embedding(*item_sz)
        self.layers = nn.Sequential(
            nn.Linear(user_sz[1]+item_sz[1], n_act),
            nn.ReLU(),
            nn.Linear(n_act, 1))
        self.y_range = y_range
        
    def forward(self, x):
        embs = self.user_factors(x[:,0]),self.item_factors(x[:,1])
        x = self.layers(torch.cat(embs, dim=1))
        return sigmoid_range(x, *self.y_range)


# In[ ]:


model = CollabNN(*embs)


# In[ ]:


learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3, wd=0.01)


# In[ ]:


learn = collab_learner(dls, use_nn=True, y_range=(0, 5.5), layers=[100,50])
learn.fit_one_cycle(5, 5e-3, wd=0.1)


# In[ ]:


@delegates(TabularModel)
class EmbeddingNN(TabularModel):
    def __init__(self, emb_szs, layers, **kwargs):
        super().__init__(emb_szs, layers=layers, n_cont=0, out_sz=1, **kwargs)


# ### Sidebar: kwargs and Delegates

# ### End sidebar

# ## Conclusion

# ## Questionnaire

# 1. What problem does collaborative filtering solve?
# 1. How does it solve it?
# 1. Why might a collaborative filtering predictive model fail to be a very useful recommendation system?
# 1. What does a crosstab representation of collaborative filtering data look like?
# 1. Write the code to create a crosstab representation of the MovieLens data (you might need to do some web searching!).
# 1. What is a latent factor? Why is it "latent"?
# 1. What is a dot product? Calculate a dot product manually using pure Python with lists.
# 1. What does `pandas.DataFrame.merge` do?
# 1. What is an embedding matrix?
# 1. What is the relationship between an embedding and a matrix of one-hot-encoded vectors?
# 1. Why do we need `Embedding` if we could use one-hot-encoded vectors for the same thing?
# 1. What does an embedding contain before we start training (assuming we're not using a pretained model)?
# 1. Create a class (without peeking, if possible!) and use it.
# 1. What does `x[:,0]` return?
# 1. Rewrite the `DotProduct` class (without peeking, if possible!) and train a model with it.
# 1. What is a good loss function to use for MovieLens? Why? 
# 1. What would happen if we used cross-entropy loss with MovieLens? How would we need to change the model?
# 1. What is the use of bias in a dot product model?
# 1. What is another name for weight decay?
# 1. Write the equation for weight decay (without peeking!).
# 1. Write the equation for the gradient of weight decay. Why does it help reduce weights?
# 1. Why does reducing weights lead to better generalization?
# 1. What does `argsort` do in PyTorch?
# 1. Does sorting the movie biases give the same result as averaging overall movie ratings by movie? Why/why not?
# 1. How do you print the names and details of the layers in a model?
# 1. What is the "bootstrapping problem" in collaborative filtering?
# 1. How could you deal with the bootstrapping problem for new users? For new movies?
# 1. How can feedback loops impact collaborative filtering systems?
# 1. When using a neural network in collaborative filtering, why can we have different numbers of factors for movies and users?
# 1. Why is there an `nn.Sequential` in the `CollabNN` model?
# 1. What kind of model should we use if we want to add metadata about users and items, or information such as date and time, to a collaborative filtering model?

# ### Further Research
# 
# 1. Take a look at all the differences between the `Embedding` version of `DotProductBias` and the `create_params` version, and try to understand why each of those changes is required. If you're not sure, try reverting each change to see what happens. (NB: even the type of brackets used in `forward` has changed!)
# 1. Find three other areas where collaborative filtering is being used, and find out what the pros and cons of this approach are in those areas.
# 1. Complete this notebook using the full MovieLens dataset, and compare your results to online benchmarks. See if you can improve your accuracy. Look on the book's website and the fast.ai forum for ideas. Note that there are more columns in the full dataset—see if you can use those too (the next chapter might give you ideas).
# 1. Create a model for MovieLens that works with cross-entropy loss, and compare it to the model in this chapter.

# In[ ]:




