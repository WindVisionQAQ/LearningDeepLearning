#!/usr/bin/env python
# coding: utf-8

# ### Week 4 project
# 要求分别建立浅层神经网络（2 layer）和深层神经网络，完成FP BP过程，最终比较两个模型的效果

# ### 库文件

# In[3]:


import numpy as np
import h5py
import matplotlib.pyplot as plt
import testCases
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
import lr_utils


# 验证结果 使用seed
np.random.seed(1)


# ### 初始化参数
# 先完成浅层神经网络的参数初始化：

# In[4]:


def init_params(n_x,n_h,n_y):
    """
    init_params用于初始化浅层NN的参数
    函数参数：
        n_x - 输入层单元数
        n_h - 隐藏层单元数
        n_y - 输出层单元数
    返回结果：
        params - 包含W1,W2,b1,b2参数的字典
    """
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))
    
    params = {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }
    return params


# 测试：

# In[5]:


print("=============Test init_params==============")
params = init_params(3,2,1)
print("W1: "+str(params["W1"]))
print("b1: "+str(params["b1"]))
print("W2: "+str(params["W2"]))
print("b2: "+str(params["b2"]))


# In[6]:


def init_params_deep(layer_dims):
    """
    参数:
        layer_dims:列表结构，存有每层的单元个数
    返回：
        params: 存有W b的字典参数
    """
    np.random.seed(3)
    params={}
    for i in range(1,len(layer_dims)):
        params["W"+str(i)] = np.random.randn(layer_dims[i],layer_dims[i-1])*0.01
        params["b"+str(i)] = np.zeros((layer_dims[i],1))
    return params


# 测试init_params_deep

# In[7]:


layer_dims = [5,4,3]
params = init_params_deep(layer_dims)
print("W1: "+str(params["W1"]))
print("b1: "+str(params["b1"]))
print("W2: "+str(params["W2"]))
print("b2: "+str(params["b2"]))


# ### Forward prop

# In[8]:


def linear_forward(A,W,b):
    Z = np.dot(W,A)+b
    cache = (A,W,b)
    return Z,cache


# 测试linear_forward

# In[9]:


A,W,b = testCases.linear_forward_test_case()
Z,linear_cache = linear_forward(A,W,b)
print("Z : "+str(Z))


# In[10]:


def linear_activation_forward(A_prev,W,b,activation):
    """
    返回：
        A - activation后的结果
        cache - 包含linear_cache(A,W,b)和activation cache(Z)
    """
    Z,linear_cache = linear_forward(A_prev,W,b)
    if activation == "sigmoid":
        A,activation_cache = sigmoid(Z)
    elif activation == "relu":
        A,activation_cache = relu(Z)
    
    cache = (linear_cache,activation_cache)
    return A,cache


# 测试Linear_activation_forward

# In[11]:


A_prev,W,b = testCases.linear_activation_forward_test_case()
A,cache = linear_activation_forward(A_prev,W,b,activation="sigmoid")
print("sigmoid A="+str(A))
A,cache = linear_activation_forward(A_prev,W,b,activation="relu")
print("relu A="+str(A))


# In[12]:


def L_model_forward(X,params):
    """
    返回：
        AL - 最后一层的输出
        caches - 包含每层的cache
    """
    caches = []
    layer_num = len(params)//2
    A = X
    for i in range(1,layer_num):
        A_prev = A
        A, cache= linear_activation_forward(A_prev,params["W"+str(i)],params["b"+str(i)],activation="relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A,params["W"+str(layer_num)],params["b"+str(layer_num)],activation="sigmoid")
    caches.append(cache)
    return AL,caches


# 测试L_model_forward

# In[13]:


X,params = testCases.L_model_forward_test_case()
AL,caches = L_model_forward(X,params)
print("AL = "+str(AL))
print("caches length: "+ str(len(caches)))


# ### 计算cost

# In[14]:


def compute_cost(AL,Y):
    m = Y.shape[1]
    temp = np.multiply(np.log(AL),Y)+np.multiply(np.log(1-AL),1-Y)
    cost = -np.sum(temp)/m
    cost = np.squeeze(cost)
    return cost


# 测试compute_cost

# In[15]:


Y,AL = testCases.compute_cost_test_case()
cost = compute_cost(AL,Y)
print("cost = "+str(cost))


# ### BP

# In[16]:


def linear_backward(dZ,cache):
    """
    linear_backward用于根据dZ[l]计算dA[l-1],dW[l],db[l]
    """
    A_prev,W,b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ,A_prev.T)/m
    db = np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev = np.dot(W.T,dZ)
    
    return dA_prev,dW,db
    


# 测试linear_backward:

# In[17]:


dZ,cache = testCases.linear_backward_test_case()
dA_prev, dW, db = linear_backward(dZ,cache)
print("dA_prev: "+str(dA_prev))
print("dW: "+str(dW))
print("db: "+str(db))


# In[18]:


def linear_activation_backward(dA,caches,activation="relu"):
    """
    参数：
        dA - 略
        caches - 包含linear_cache和activation_cache
        activation - 略
    返回：
        dA_prev, dW, db
    """
    linear_cache, activation_cache = caches
    if activation=="relu":
        dZ = relu_backward(dA,activation_cache)
    elif activation=="sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
    return linear_backward(dZ,linear_cache)


# In[19]:


#测试linear_activation_backward
print("==============测试linear_activation_backward==============")
AL, linear_activation_cache = testCases.linear_activation_backward_test_case()
 
dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")
print ("sigmoid:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db) + "\n")
 
dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "relu")
print ("relu:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))


# In[20]:


def L_model_backward(AL,Y,caches):
    """
    返回：
        grads - 包含各层dW，db的字典
    """
    grads = {}
    dAL = -(np.divide(Y,AL) - np.divide(1-Y,1-AL))
    grads["dAL"] = dAL
    L = len(caches)
    dA_prev,dW,db= linear_activation_backward(dAL,caches[L-1],activation="sigmoid")
    grads["dW"+str(L)] = dW
    grads["db"+str(L)] = db
    grads["dA"+str(L-1)] = dA_prev
    i = L-1
    while(i>=1):
        dA = dA_prev
        dA_prev,dW,db = linear_activation_backward(dA,caches[i-1],activation="relu")
        grads["dW"+str(i)] = dW
        grads["db"+str(i)] = db
        grads["dA"+str(i-1)] = dA_prev
        i = i - 1
    return grads


# In[21]:


#测试L_model_backward
print("==============测试L_model_backward==============")
AL, Y_assess, caches = testCases.L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dA1 = "+ str(grads["dA1"]))


# ### Gradient Descent

# In[22]:


def update_params(params,grads,learning_rate):
    layer_num = len(params) // 2
    for i in range(1,layer_num+1):
        params["W"+str(i)] = params["W"+str(i)] - learning_rate * grads["dW"+str(i)]
        params["b"+str(i)] = params["b"+str(i)] - learning_rate * grads["db"+str(i)]
    return params    


# 测试梯度下降

# In[24]:


#测试update_parameters
print("==============测试update_parameters==============")
parameters, grads = testCases.update_parameters_test_case()
parameters = update_params(parameters, grads, 0.1)
 
print ("W1 = "+ str(parameters["W1"]))
print ("b1 = "+ str(parameters["b1"]))
print ("W2 = "+ str(parameters["W2"]))
print ("b2 = "+ str(parameters["b2"]))


# In[32]:


def two_layer_model(X,Y,layer_dims,learning_rate=0.0075,num_iterations=3000,print_cost=False,isPlot=True):
    np.random.seed(1)
    grads={}
    costs=[]
    (n_x,n_h,n_y) = layer_dims
    
    params = init_params(n_x,n_h,n_y)
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    
    for i in range(num_iterations):
        A1,cache1 = linear_activation_forward(X,W1,b1,"relu")
        A2,cache2 = linear_activation_forward(A1,W2,b2,"sigmoid")
        
        cost = compute_cost(A2,Y)
        
        dA2 = -(np.divide(Y,A2) - np.divide(1-Y,1-A2))
        dA1,dW2,db2 = linear_activation_backward(dA2,cache2,"sigmoid")
        dA0,dW1,db1 = linear_activation_backward(dA1,cache1,"relu")
        
        grads = {
            "dW1":dW1,
            "db1":db1,
            "dW2":dW2,
            "db2":db2
        }
        params = update_params(params,grads,learning_rate)
        W1 = params["W1"]
        W2 = params["W2"]
        b1 = params["b1"]
        b2 = params["b2"]
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("Iteration "+str(i)+" cost:"+str(cost))
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel("cost")
        plt.xlabel("#Iteration")
        plt.title("Learning Rate = "+str(learning_rate))
        plt.show()
    return params
    
        
        


# ### 加载数据

# In[33]:


train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = lr_utils.load_dataset()

train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T 
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y


# ### 使用两层网络训练

# In[34]:


n_x = 12288
n_h = 7
n_y = 1
layers_dims = (n_x,n_h,n_y)

parameters = two_layer_model(train_x, train_set_y, layers_dims,0.0075,num_iterations = 2500, print_cost=True,isPlot=True)



# ### 预测

# In[35]:


def predict(X,y,params):
    m = X.shape[1]
    p = np.zeros((1,m))
    
    probas,caches = L_model_forward(X,params)
    
    for i in range(m):
        if(probas[0,i]>0.5):
            p[0,i] = 1
        else:
            p[0,i] = 0
    acc = (np.sum(p==y))/m
    print("Accuracy :"+str(acc))
    return p


# In[36]:


predictions_train = predict(train_x, train_y, parameters) #训练集
predictions_test = predict(test_x, test_y, parameters) #测试集


# ### 多层神经网络

# In[51]:


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False,isPlot=True):

    np.random.seed(1)
    costs = []
    
    parameters = init_params_deep(layers_dims)
    
    for i in range(0,num_iterations):
        AL , caches = L_model_forward(X,parameters)
        
        cost = compute_cost(AL,Y)
        
        grads = L_model_backward(AL,Y,caches)
        
        parameters = update_params(parameters,grads,learning_rate)
        

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("第", i ,"次迭代，成本值为：" ,np.squeeze(cost))

    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    return parameters


# ### 加载数据
# 
# 

# In[57]:


train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = lr_utils.load_dataset()

train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T 
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y

print(train_x.shape)
print(train_x)


# ### 训练模型

# In[58]:


layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
parameters = L_layer_model(train_x, train_y, layers_dims, 0.0075,num_iterations = 2500,print_cost = True,isPlot=True)


# In[ ]:




