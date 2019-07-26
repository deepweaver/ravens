import numpy as np 
import matplotlib.pyplot as plt 
import pickle 
from utils import * 
from Analyzer import CellAnalyzer 
import tqdm 





class GestaltAnalyzer(CellAnalyzer): 
    def __init__(self, mat): 
        CellAnalyzer.__init__(self, mat) 
        self.featureMatchingFunc = FeatureMatcher.SIFTdetectorAndFLANNmatcher
        self.a = CellAnalyzer(self.mat[              :self.height//2,              :self.width//2]) 
        self.b = CellAnalyzer(self.mat[              :self.height//2, self.width//2:             ])
        self.c = CellAnalyzer(self.mat[self.height//2:              ,              :self.width//2]) 
        self.d = CellAnalyzer(self.mat[self.height//2:              , self.width//2:             ])
        
    def getAllScores(self,): 
        scores = []
        # scores.append(self.verticalSimilarityScore()) 
        # scores.append(self.verticalSymmetryScore()) 
        # scores.append(self.horizontalSymmetryScore()) 
        # scores.append(self.horizontalSimilarityScore()) 
        scores.append(self.progressionScore()) 
        return scores

    def verticalSymmetryScore(self, ): 
        leftHalf = self.mat[:,:self.width//2] 
        rightHalf = self.mat[:,self.width//2:]
        return self.featureMatchingFunc(leftHalf, np.flip(rightHalf,1)) 

    def horizontalSymmetryScore(self, ): 
        upperHalf = self.mat[:self.height//2,:] 
        lowerHalf = self.mat[self.height//2:,:] 
        return self.featureMatchingFunc(upperHalf, np.flip(lowerHalf,1)) 

    def verticalSimilarityScore(self, ): 
        leftHalf = self.mat[:,:self.width//2] 
        rightHalf = self.mat[:,self.width//2:]
        return self.featureMatchingFunc(leftHalf, rightHalf) 

    def horizontalSimilarityScore(self,): 
        upperHalf = self.mat[:self.height//2,:] 
        lowerHalf = self.mat[self.height//2:,:] 
        return self.featureMatchingFunc(upperHalf, lowerHalf)

    def continuityScore(self,):  # failed 
        edges = cv2.Canny(self.mat, 100, 200) 
        return (edges==255).sum() 

    def linearTotalScore(self,): # failed 
        scores = np.array(self.getAllScores() + [1] )
        w = np.array([-7.46565552e+01, -7.87536392e+01, -7.78749466e+01, -7.51255875e+01, -1.66666554e-03]) 
        return sigmoid(np.dot(scores,w))

    def progressionScore(self,): #  all non negative, the closer to 0 the better 
        # (d-c) - (b-a) 
        # return np.abs((self.d.numberOfBlackPixels-self.c.numberOfBlackPixels)-(self.b.numberOfBlackPixels-self.a.numberOfBlackPixels))
        
        try:
            peri = np.abs((self.d.perimeter-self.c.perimeter)-(self.b.perimeter-self.a.eprimeter))
        except: 
            print("err")
        else:
            peri = 1000 
        finally: 
            return peri


if __name__ == "__main__": 
    correctAnwsers = [4,5,1,2,6,3,6,2,1,3,4,5,2,6,1,2,1,3,5,6,4,3,4,5] 

    with open("../data/data_a_b_570x900_py3.pkl", "rb") as file: 
        data_a_b = pickle.load(file) 
    
    ans = []
    data_saver = []
    for k in tqdm.tqdm(range(24)): 
        tmp = [] 
        for i in range(6): 
            
            analyzer = GestaltAnalyzer(data_a_b[k][i,:,:]) 
            # tmp.append(analyzer.getAllScores()) # get [4 scores] of each pic
            tmp.append(sum(analyzer.getAllScores())) 
            # tmp.append(analyzer.continuityScore()) 
            # tmp.append(analyzer.linearTotalScore()) 

        ans.append(np.array(tmp).argmin()+1) 
        # ans.append(tmp)
    # with open("../data/first24Scores.pkl", "wb") as file: 
    #     pickle.dump(ans,file)
    print(correctAnwsers)
    print(ans) 

    correct_cnt = 0
    for i in range(24): 
        if ans[i] == correctAnwsers[i]: 
            correct_cnt += 1 
    print("current count = {}".format(correct_cnt)) 















