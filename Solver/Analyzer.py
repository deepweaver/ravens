from PIL import Image, ImageChops 
import numpy as np 
import cv2 
# cv
# import copy 
from utils import visualize 

class CellAnalyzer(object): 
    def __init__(self, mat):
        # image is a numpy array
        self.mat = mat 
        self.numberOfWhitePixels = np.sum(self.mat/255) 
        self.numberOfBlackPixels = np.prod(self.mat.shape) - self.numberOfWhitePixels
        self.blackWhiteRatio = 1.0 * self.numberOfBlackPixels / self.numberOfWhitePixels if self.numberOfWhitePixels != 0 else 0
        # 
        self.perimeter = 0 
        self.centroid = 0 
        self.contour = self.getContour()
        self.ymin, self.ymax, self.xmin, self.xmax = self.getBoundingBox()
        self.width, self.height = (self.xmax - self.xmin), (self.ymax - self.ymin) 
        self.alignedContour = self.getAlignedContour()
        self.alignedMat = self.getAlignedMat() 
        # self.contour is a numpy array of shape [numOfPoints, 2, 1]

        # print(len(self.contours))
        # print(self.contours[0].shape)
    # def getBoundingBox(self):
    #     return 0, 0
    def getContour(self): 
        contours = cv2.findContours(255-self.mat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        # contours = cv2.findContours(255-self.mat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        # print("length of contours is " , len(contours))
        if len(contours) > 1: 
            IdxOfContourWithMaxLength = 0 
            Max = 0 
            for i, cts in enumerate(contours): 
                if cts.shape[0] > Max: 
                    Max = cts.shape[0] 
                    IdxOfContourWithMaxLength = i
            return contours[IdxOfContourWithMaxLength]
        elif len(contours) == 1:
            return contours[0]
        else: 
            # print("can't find contours in this image")
            return None 
    def getAlignedContour(self):
        if self.contour is not None:
            return self.contour - np.min(self.contour,axis=0)
        else:
            return None 
    
    def getAlignedMat(self):
        if self.contour is not None:
            alignedMat = 255*np.ones(self.mat.shape, dtype=np.uint8) 
            # x, y = np.min(self.contour, axis=0)[0,:]
            # w, h = self.width, self.height 
            alignedMat[0:self.height, 0:self.width] = self.mat[self.ymin:self.ymin+self.height,self.xmin:self.xmin+self.width]  
            return alignedMat 
        else: 
            return self.mat  # nothing to aligned since there are nothing in the picture

    def getBoundingBox(self):

        if self.contour is None: 
            return (0, self.mat.shape[0], 0, self.mat.shape[1]) 
        else:
            img = 1 - self.mat/255 
            rows = np.any(img, axis=1)
            cols = np.any(img, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            return (rmin, rmax, cmin, cmax) 




def instantializeListOfAnalyzers(mats):
    return [ CellAnalyzer(mat) for mat in mats ] 

def visualizeContours(matAnalyzers, save=False, show=True): # input is a list of numpy arrays 
    # matAnalyzers = copy.deepcopy(matAnalyzers) 
    mats2beVisualized = [] 
    for matAnalyzer in matAnalyzers: 
        tmpMat = matAnalyzer.mat.copy() 
        if matAnalyzer.contour is not None:
            cv2.drawContours(tmpMat, [matAnalyzer.contour], -1, (124,252,0), 5) 
        mats2beVisualized.append(tmpMat) 
    visualize(mats2beVisualized, save=save, show=show) 


def measureContourSimilarity(mat1Analyzer, mat2Analyzer):
    '''
        input two numpy array 
        compute contours 
        measure similarity 
        return 
            1 -- similar
            0 -- not similar 
            -1 -- symmetry 
    '''
    if mat1Analyzer.contour is None or mat2Analyzer.contour is None: 
        # print("contour not found")
        return 
    hd = cv2.createHausdorffDistanceExtractor()
    # sd = cv2.createShapeContextDistanceExtractor()
    d1 = hd.computeDistance(mat1Analyzer.contour, mat2Analyzer.contour)
    d2 = hd.computeDistance(mat1Analyzer.alignedContour, mat2Analyzer.alignedContour)
    # d2 = sd.computeDistance(mat1Analyzer.contour, mat2Analyzer.contour)
    # similarityValue = cv2.ShapeDistanceExtractor.computeDistance([mat1Analyzer.contour], [mat2Analyzer.contour])
    # similarityValue = cv2.HausdorffDistanceExtractor.computeDistance(mat1Analyzer.contour, mat2Analyzer.contour)
    # print(d1, d2)

def measureHistogramSimilarity(mat1Analyzer, mat2Analyzer): # not very useful
    # print(mat1Analyzer.numberOfBlackPixels, mat2Analyzer.numberOfBlackPixels)
    if np.abs(mat1Analyzer.numberOfBlackPixels - mat2Analyzer.numberOfBlackPixels) < 200: 
        return 1 
    else:
        return 0 


def measureL2Similarity(mat1Analyzer, mat2Analyzer): # utterly useless
    retval = np.sum(np.abs(mat1Analyzer.mat - mat2Analyzer.mat)) 
    # print(retval) 
    return retval 

def measureHorizontalL1Similarity(mat1Analyzer, mat2Analyzer): 
    retval = np.sum(np.abs(np.sum(mat1Analyzer.mat, axis=1) - np.sum(mat2Analyzer.mat, axis=1))) 
    return retval 

def findIdenticalPattern4cells(listOfMatAnalyzers): 
    threshold = 1000
    if np.abs(listOfMatAnalyzers[0].numberOfBlackPixels - listOfMatAnalyzers[1].numberOfBlackPixels) < 1000 and np.abs(listOfMatAnalyzers[1].numberOfBlackPixels - listOfMatAnalyzers[2].numberOfBlackPixels) < 1000 and np.abs(listOfMatAnalyzers[0].numberOfBlackPixels - listOfMatAnalyzers[2].numberOfBlackPixels) < 1000: 
        pass 
    else:
        print("first three items not identical") 
        return 0 
    averageBlackPixelsOfFirst3Cells = (listOfMatAnalyzers[0].numberOfBlackPixels + listOfMatAnalyzers[1].numberOfBlackPixels + listOfMatAnalyzers[2].numberOfBlackPixels)/3
    qualifiedCellIdx = [] 
    for i in range(4,10):
        if np.abs(listOfMatAnalyzers[0].numberOfBlackPixels - listOfMatAnalyzers[i].numberOfBlackPixels) < 800: 
            qualifiedCellIdx.append(i) 
    print("passed identity test")
    if len(qualifiedCellIdx) == 1: 
        return qualifiedCellIdx[0] - 3
    
    elif len(qualifiedCellIdx) > 1: # find more than one answer by this measure 
        print("find more than one answer")
        # print(qualifiedCellIdx)
        idxOfCellWithLeastSimilarity = 0 
        leastSimilarity = 10000000
        for j, v in enumerate(qualifiedCellIdx):
            sim = measureHorizontalL1Similarity(listOfMatAnalyzers[0], listOfMatAnalyzers[qualifiedCellIdx[j]])
            # print(sim)
            if sim < leastSimilarity: 
                leastSimilarity = sim 
                idxOfCellWithLeastSimilarity = v 
        
        return idxOfCellWithLeastSimilarity - 3
    else: # len(qualifiedCellIdx) == 0 can't find identical cell by this measure 
        print("can't find any answer")
        return 0



def main():
    import pickle
    

    with open("data.pkl", 'rb') as f: 
        dataset = pickle.load(f) 
    # for i in range(10):
    #     data = dataset["4cells"][i]
    #     print("This is the {}".format(i)) 
    #     dataAnalyzers = instantializeListOfAnalyzers(data)  
    #     visualizeContours(dataAnalyzers, save=True)
    #     # print(len(dataAnalyzers[5].contours))
    #     # print(type(dataAnalyzers[5].contours[0]))
    #     # print(dataAnalyzers[5].contour)

    #     # print(dataAnalyzers[0].contour) 
    #     # print(dataAnalyzers[0].alignedContour)

    #     for analyzer in dataAnalyzers: 
    #         # measureContourSimilarity(dataAnalyzers[0], analyzer)
    #         # measureL2Similarity(dataAnalyzers[0], analyzer)
    #         measureHistogramSimilarity(dataAnalyzers[0], analyzer)
    #         # print(measureHistogramSimilarity(dataAnalyzers[0], analyzer))
    #         # measureL2Similarity(dataAnalyzers[0], analyzer)
    #         # print(analyzer.width, analyzer.height)
    #     print("----------------------------")

    correctAnwsers = [4,5,1,2,6,3,6,2,1,3,4,5,2,6,1,2,1,3,5,6,4,3,4,5] 
    answers = [] 
    for i in range(0,24):
        print("index = {}".format(i)) 
        data = dataset["4cells"][i]
        dataAnalyzers = instantializeListOfAnalyzers(data) 
        # retval = findIdenticalPattern4cells(dataAnalyzers)
        # visualize(data, answer=retval, title=str(i), save=True, show=True)
        alignedData = [] 
        
        for a in dataAnalyzers: 
            alignedData.append(a.alignedMat) 
        answers.append(findIdenticalPattern4cells(instantializeListOfAnalyzers(alignedData)))
        # visualize(alignedData)
    correctCount = 0 
    for i, v in enumerate(answers): 
        if correctAnwsers[i] == answers[i]: 
            correctCount += 1 
    print(correctAnwsers) 
    print(answers)
    print("{} question answered correctly".format(correctCount)) 



if __name__ == "__main__":
    main()












