
# coding: utf-8

# In this problem we apply PCA and kernel PCA for learning acoustic features and plot the top two princi-  pal components as a tool for visualizing articulatory information captured in acoustic features. Fetch the  baseline acoustic features and corresponding vowel labels  { ‘AE’, ‘AO’, ‘OW’, ‘UW’, ‘IY’, ‘AA’, ‘EH’ }. The baseline features are 39-dimensional MFCCs (mel-frequency cepstral coefficients). The dataset is split into a training set, a tuning  set (or dev set) and a test set. 
# 
# # (a)
# Learn principal directions on the training data and print the scatter plot of the projection of both training  data and the test data on to the top two principal directions. We can group vowels into “closed”  {‘  UW’,  ‘IY’  }  and “open”  { ‘AE’, ‘AA’  }  based on shape of mouth at articulation time. Does PCA preserve  the separation between these two groups? 

# In[3]:

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# In[4]:

d = sio.loadmat('./XRMBsJW11.mat')


# In[5]:

print np.unique(d.keys())


# In[6]:

len_train = np.size(d['Phones']['training'][0][0])
ae_index = []
ao_index = []
ow_index = []
uw_index = []
iy_index = []
aa_index = []
eh_index = []
for i in xrange(len_train):
    if d['Phones']['training'][0][0][:,i] == '"AE"':
        ae_index.append(i)
    elif d['Phones']['training'][0][0][:,i] == '"AO"':
        ao_index.append(i)
    elif d['Phones']['training'][0][0][:,i] == '"OW"':
        ow_index.append(i)
    elif d['Phones']['training'][0][0][:,i] == '"UW"':
        uw_index.append(i)
    elif d['Phones']['training'][0][0][:,i] == '"IY"':
        iy_index.append(i)
    elif d['Phones']['training'][0][0][:,i] == '"AA"':
        aa_index.append(i)
    elif d['Phones']['training'][0][0][:,i] == '"EH"':
        eh_index.append(i)
        


# In[7]:

print np.shape(ae_index)
print np.shape(ao_index)
print np.shape(uw_index)
print np.shape(iy_index)
print np.shape(aa_index)
print np.shape(eh_index)


# In[8]:

data_train = d['data']['training'][0][0]     # (features,points)
print np.shape(data_train)


# In[9]:

ae_data = data_train[:,ae_index]
ao_data = data_train[:,ao_index]
uw_data = data_train[:,uw_index]
iy_data = data_train[:,iy_index]
aa_data = data_train[:,aa_index]
eh_data = data_train[:,eh_index]


# In[10]:

print np.shape(ae_data)
print np.shape(ao_data)
print np.shape(uw_data)
print np.shape(iy_data)
print np.shape(aa_data)
print np.shape(eh_data)


# In[11]:

closed_data_index = uw_index + iy_index
open_data_index = ae_index + aa_index
print np.shape(closed_data_index)
print np.shape(open_data_index)


# In[12]:

pca = PCA(n_components=2)
data_proj = pca.fit_transform(np.transpose(data_train))
print np.shape(data_proj)
closed_proj = data_proj[closed_data_index,:]
open_proj = data_proj[open_data_index,:]
print np.shape(closed_proj)
print np.shape(open_proj)


# In[13]:

plt.scatter(closed_proj[:,0],closed_proj[:,1],c='r',marker='o')
plt.scatter(open_proj[:,0],open_proj[:,1],c='b',marker='o')
plt.show()


# # (b)
# A polynomial kernel is given as  k (x ,  y) = (  x   ,  y   +  c )α    for some offset  c  ≥  0 and degree  α  ∈  Z.  Describe  the feature map Φ :  X → H  corresponding to the polynomial kernel with degree  α  = 2 and offset  c  = 2.  Perform kernel PCA using the polynomial kernel ( α  = 2,  c  = 2) by explicitly computing the feature map  and performing PCA in the feature space. Generate the scatter plot of the projections of both training  data and the test data on to the top two kernel principal directions. What is the dimensionality of the  feature map associated with a polynomial kernel of degree  α  with  d -dimensional input features? 

# In[27]:

from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import polynomial_kernel


# In[28]:

# data_train_poly = polynomial_kernel(np.transpose(data_train), degree=2, coef0=2,gamma=None)
# print np.shape(data_train_poly)
# pca = PCA(n_components=2)
# data_proj = pca.fit_transform(np.transpose(data_train_poly))
# print np.shape(data_proj)
# ae_proj = data_proj[ae_index,:]
# ao_proj = data_proj[ao_index,:]
# uw_proj = data_proj[uw_index,:]
# iy_proj = data_proj[iy_index,:]
# aa_proj = data_proj[aa_index,:]
# eh_proj = data_proj[eh_index,:]


# In[34]:

kpca = KernelPCA(kernel="poly",coef0=2,degree=2,n_components=2)
data_train = np.transpose(data_train)
data_proj = kpca.fit_transform(data_train)
print np.shape(data_proj)
ae_proj = data_proj[ae_index,:]
ao_proj = data_proj[ao_index,:]
uw_proj = data_proj[uw_index,:]
iy_proj = data_proj[iy_index,:]
aa_proj = data_proj[aa_index,:]
eh_proj = data_proj[eh_index,:]


# In[35]:

plt.scatter(ae_proj[:,0],ae_proj[:,1],c='r',marker='o')
plt.scatter(ao_proj[:,0],ao_proj[:,1],c='g',marker='o')
plt.scatter(uw_proj[:,0],uw_proj[:,1],c='b',marker='o')
plt.scatter(iy_proj[:,0],iy_proj[:,1],c='y',marker='o')
plt.scatter(aa_proj[:,0],aa_proj[:,1],c='c',marker='o')
plt.scatter(eh_proj[:,0],eh_proj[:,1],c='m',marker='o')
plt.show()


# In[36]:

plt.scatter(data_proj[:,0],data_proj[:,1],c='r',marker='o')
plt.show()


# # (c)
# Perform kernel PCA using a Gaussian kernel  k (x , y) = exp −  x − y 2  2 
# σ 2  using a kernel bandwidth  σ  = 30  on the training data and generate the scatter plot of projections of the training data and the test data  on to the top two kernel principal directions. You may randomly subsample the data if computing,  storing and factorizing the full kernel matrix becomes computationally infeasible or intensive on your 
#  local machine.

# In[14]:

from sklearn.metrics.pairwise import rbf_kernel


# In[15]:

print np.shape(data_train)


# In[31]:

data_train_rbf = rbf_kernel(np.transpose(data_train),Y=np.transpose(data_train), gamma=1/30)
print np.shape(data_train_rbf)
pca = PCA(n_components=2)
data_proj = pca.fit_transform(np.transpose(data_train_rbf))
print np.shape(data_proj)
ae_proj = data_proj[ae_index,:]
ao_proj = data_proj[ao_index,:]
uw_proj = data_proj[uw_index,:]
iy_proj = data_proj[iy_index,:]
aa_proj = data_proj[aa_index,:]
eh_proj = data_proj[eh_index,:]


# In[32]:

plt.scatter(ae_proj[:,0],ae_proj[:,1],c='r',marker='o')
plt.scatter(ao_proj[:,0],ao_proj[:,1],c='g',marker='o')
plt.scatter(uw_proj[:,0],uw_proj[:,1],c='b',marker='o')
plt.scatter(iy_proj[:,0],iy_proj[:,1],c='y',marker='o')
plt.scatter(aa_proj[:,0],aa_proj[:,1],c='c',marker='o')
plt.scatter(eh_proj[:,0],eh_proj[:,1],c='m',marker='o')
plt.show()


# In[17]:

from sklearn.decomposition import KernelPCA
X = np.transpose(data_train)
kpca = KernelPCA(kernel="rbf",gamma=1/30,n_components=2)
print np.shape(X)


# In[18]:

X_kpca = kpca.fit_transform(X)


# In[19]:

data_proj = X_kpca
ae_proj = data_proj[ae_index,:]
ao_proj = data_proj[ao_index,:]
uw_proj = data_proj[uw_index,:]
iy_proj = data_proj[iy_index,:]
aa_proj = data_proj[aa_index,:]
eh_proj = data_proj[eh_index,:]
print np.shape(ae_proj)
plt.scatter(ae_proj[:,0],ae_proj[:,1],c='r',marker='o')
plt.scatter(ao_proj[:,0],ao_proj[:,1],c='g',marker='o')
plt.scatter(uw_proj[:,0],uw_proj[:,1],c='b',marker='o')
plt.scatter(iy_proj[:,0],iy_proj[:,1],c='y',marker='o')
plt.scatter(aa_proj[:,0],aa_proj[:,1],c='c',marker='o')
plt.scatter(eh_proj[:,0],eh_proj[:,1],c='m',marker='o')
plt.show()


# In[ ]:



