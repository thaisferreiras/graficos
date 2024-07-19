#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install -q seaborn==0.12.2')


# In[3]:


# imports
import random
import numpy as np
import pandas as pd
import matplotlib as mat
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


import seaborn as sea


# In[5]:


# carregando um dos datasets que vem com o seaborn
dados = sea.load_dataset("tips")


# In[6]:


dados.head()


# In[10]:


# o metodo joinplot cria plot de 2 variaveis com graficos bivariados e univariados
sea.jointplot(data = dados, x = "total_bill", y = "tip", kind = 'reg')


# In[11]:


# o metodo lmplot cria plot com dados e modelos de regressao
sea.lmplot(data = dados, x = "total_bill", y = "tip", col = 'smoker')


# In[12]:


# construindo um dataframe com pandas
df = pd.DataFrame()


# In[13]:


# alimentando o dataframe com valores aleatórios
df['idade'] = random.sample(range(20,100),30)
df['peso'] = random.sample(range(55,150),30)


# In[14]:


df.shape


# In[15]:


df.head()


# In[16]:


# lmplot
sea.lmplot(data = df, x = "idade", y = "peso", fit_reg = True)


# In[18]:


# Kdeplot
sea.kdeplot(df.idade)


# In[19]:


# Kdeplot
sea.kdeplot(df.peso)


# In[20]:


# distplot
sea.distplot(df.peso)


# In[21]:


# histograma
plt.hist(df.idade, alpha = .3)
sea.rugplot(df.idade)


# In[22]:


# box plot
sea.boxplot(df.idade, color = 'm')


# In[23]:


# box plot
sea.boxplot(df.peso, color = 'y')


# In[24]:


# violin plot
sea.violinplot(df.idade, color = 'g')


# In[25]:


# violin plot
sea.violinplot(df.peso, color = 'cyan')


# In[27]:


sea.clustermap(df)


# #  usando Matplotlib, seaborn, numpy e pandas para criação de graficos

# In[30]:


# valores randomicos
np.random.seed(42)
n = 1000
pct_smokers = 0.2

#variaveis
flag_fumante = np.random.rand(n) < pct_smokers
idade = np.random.normal(40, 10, n)
altura = np.random.normal(170, 10, n)
peso = np.random.normal(70, 10, n)

# dataframes
dados = pd.DataFrame({'altura': altura, 'peso': peso, 'flag_fumante': flag_fumante})

# cria os dados para a variavel flag_fumante
dados['flag_fumante'] = dados['flag_fumante'].map({True: 'Fumante', False: 'Não Fumante'})


# In[31]:


dados.shape


# In[32]:


dados.head()


# In[34]:


# style
sea.set(style = "ticks")

#lmplot
sea.lmplot(x = 'altura',
           y = 'peso',
           data = dados,
           hue = 'flag_fumante',
           palette = ['tab:blue', 'tab:orange'],
           height = 7)

# Labels e Titulos
plt.xlabel('Altura (cm)')
plt.ylabel('Peso (Kg)')
plt.title('Relação de Altura e Peso de Fumante e Não Fumante')

# remove as bordas
sea.despine()

# show
plt.show()


# In[ ]:




