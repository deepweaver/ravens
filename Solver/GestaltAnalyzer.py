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
        
    def getAllScores(self,): 
        scores = []
        scores.append(self.verticalSimilarityScore()) 
        scores.append(self.verticalSymmetryScore()) 
        scores.append(self.horizontalSymmetryScore()) 
        scores.append(self.horizontalSimilarityScore()) 
        return scores


    def verticalSymmetryScore(self, ): 
        leftHalf = self.mat[:,:self.width//2] 
        rightHalf = self.mat[:,self.width//2:]
        return self.featureMatchingFunc(leftHalf, np.flip(rightHalf,1)) 
        # return ORBdetectorAndBFmatcher(leftHalf, np.flip(rightHalf,1)) 
        # return 0 

    def horizontalSymmetryScore(self, ): 
        upperHalf = self.mat[:self.height//2,:] 
        lowerHalf = self.mat[self.height//2:,:] 
        return self.featureMatchingFunc(upperHalf, np.flip(lowerHalf,1)) 
        # return ORBdetectorAndBFmatcher(upperHalf, np.flip(lowerHalf,1)) 
        # return 0 

    def verticalSimilarityScore(self, ): 
        leftHalf = self.mat[:,:self.width//2] 
        rightHalf = self.mat[:,self.width//2:]
        return self.featureMatchingFunc(leftHalf, rightHalf) 
        # return ORBdetectorAndBFmatcher(leftHalf, rightHalf) 

    def horizontalSimilarityScore(self,): 
        upperHalf = self.mat[:self.height//2,:] 
        lowerHalf = self.mat[self.height//2:,:] 
        return self.featureMatchingFunc(upperHalf, lowerHalf)
        # return ORBdetectorAndBFmatcher(upperHalf, lowerHalf)

    def continuityScore(self,): 
        edges = cv2.Canny(self.mat, 100, 200) 
        return (edges==255).sum() 

    def linearTotalScore(self,):
        pass 




if __name__ == "__main__": 
    correctAnwsers = [4,5,1,2,6,3,6,2,1,3,4,5,2,6,1,2,1,3,5,6,4,3,4,5] 

    with open("../data/data_a_b_570x900_py3.pkl", "rb") as file: 
        data_a_b = pickle.load(file) 
    
    ans = []
    for k in tqdm.tqdm(range(24)): 
        tmp = [] 
        for i in range(6): 
            
            analyzer = GestaltAnalyzer(data_a_b[k][i,:,:]) 
            
            tmp.append(sum(analyzer.getAllScores())) 
            # tmp.append(analyzer.continuityScore()) 
        print(tmp) 
        ans.append(np.array(tmp).argmin()+1)
    print(correctAnwsers)
    print(ans) 

    correct_cnt = 0
    for i in range(24): 
        if ans[i] == correctAnwsers[i]: 
            correct_cnt += 1 
    print("current count = {}".format(correct_cnt)) 















