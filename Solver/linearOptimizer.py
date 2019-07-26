import pickle 
import numpy as np 
import torch 
import copy 
from utils import get_one_hot 
import tqdm 
import matplotlib.pyplot as plt 


with open("../data/first24Scores.pkl", "rb") as file: 
    data = pickle.load(file) 

print(len(data)) # 24
print(len(data[0])) # 6
print(len(data[0][0])) # 4 
correctAnwsers = np.array([4,5,1,2,6,3,6,2,1,3,4,5,2,6,1,2,1,3,5,6,4,3,4,5])
correctAnwsers_onehot = torch.tensor(get_one_hot(correctAnwsers-1,6)).double() # return shape (24,6)


class Func():
    def __init__(self, w): 
        self.w = w 
    def forward(self, data): 
        self.w = torch.tensor(self.w, dtype=torch.float, requires_grad=True)  
        Y = torch.tensor([], requires_grad=True) 
        for i in range(24): 
            # print(i)
            outs = torch.tensor([], requires_grad=True)  
            for j in range(6): 
                tmp = copy.deepcopy(data[i][j] )
                tmp.append(1) 
                x = torch.tensor(tmp) # convert it to tensor, input data does not requre gradient

                outs = torch.cat([outs, torch.dot(x, self.w).reshape((1,1))], 1) 
            y = torch.sigmoid(outs) 
            assert y.shape == torch.Size([1,6]) 
            # y = (torch.argmin(outs).float() + 1 ) not differenciable

            Y = torch.cat([Y, y],0)
        assert Y.shape == torch.Size([24,6]) 
        return Y 

    def getGrad(self): 
        return self.w.grad 


lr = 0.01 
# done = False 
w = np.array([0.,0.,0.,0.,0.])
losses = [] 
for i in tqdm.tqdm(range(1000)):

    f = Func(w) 
    Y = f.forward(data) 
    mseloss = torch.nn.MSELoss() 

    loss = mseloss(correctAnwsers_onehot, Y.double())
    losses.append(loss.item())
    # loss = torch.sum(torch.abs(correctAnwsers-Y)) 
    # loss = torch.nn.functional.MSELoss()
    loss.backward() 
    grad = f.getGrad() 
    w -= lr*grad.numpy()


plt.plot(losses) 
plt.show()
print(w)

print(losses[-1])


















