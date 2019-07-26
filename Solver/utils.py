from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
import datetime,time
import os 
import cv2 


def visualize(data, answer=0, title=None, save=False, show=True):
    # input a list of sub images 
    # e.g. 
    # dataset["4cells"][23] 
    # dataset["9cells"][47] 
    # output an image presentation 
    fig = plt.figure(figsize=(7,15)) 
    if title:
        fig.suptitle(title, fontsize=14, y=0.97, fontweight='semibold')
    if len(data) == 10:  
        columns = 3 
        rows = 4 
    elif len(data) == 17:
        columns = 4 
        rows = 5 
    else:
        print("Data error") 

    arrIdx = 0 
    for i in range(1,columns*rows+1): 
        if i == columns or i == 2*columns or len(data) == 17 and i == 3*columns: 
            continue 

        img = data[arrIdx]
        arrIdx += 1 
        tmp = fig.add_subplot(rows, columns, i)
        if len(data) == 10 and 0 < answer <= 6: 
            if i == 6 + answer: 
                tmp.title.set_text('Answer:') 
        if len(data) == 17 and 0 < answer <= 8: 
            if i == 12 + answer: 
                tmp.title.set_text('Answer:') 
        plt.setp(tmp.get_xticklabels(), visible=False)
        plt.setp(tmp.get_yticklabels(), visible=False)
        tmp.tick_params(axis='both', which='both', length=0)
        
        plt.imshow(img,cmap='gray')
    if save == True: 
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
        if not os.path.exists("./output/"):
            os.mkdir("./output/")
            print("Made new dir for output")
        plt.savefig("./output/{}.png".format(st)) 
    if show == True:
        plt.show()

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])
def sigmoid(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+np.exp(-x))


class ContourManager(object): 
    def __init__(self, mat): 
        self.mat = mat # input image should be binary 
        _, self._contours, self._hierarchy = cv2.findContours(255-self.mat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # hierarchy is a numpy array of shape (1, self.numOfContours, 4) 4 being [Next, Previous, First_Child, Parent]
        # contours is a list of numpy arrays with shape (number of points, 1, x and y position) 
        self.numOfContours = len(self._contours)
        self._computePerimeters() 

    def getAllContours(self, ): 
        return self._contours 

    def getLv1Contours(self, ): 
        self._computeLv1Contours() 
        return self.lv1Cnt 

    @staticmethod 
    def showContours(contours, cntIdx=-1, save=False): 
        # contours is a list of numpy arrays with shape (number of points, 1, x and y position)
        # cntIdx being the index, -1 if choose all of them 
        img = self.mat.copy() 
        cv2.drawContours(img, contours, cntIdx, (128,255,0), 2) 
        plt.implot(img) 
        plt.show() 

    def getPerimeters(self): 
        return self._perimeters 
    def _computePerimeters(self,): 
        self._perimeters = [] 
        for i in range(self.numOfContours):
            self._perimeters.append(cv2.arcLength(self._contours[i], True)) 
    def _computeLv1Contours(self, ):
        self.lv1Cnt = [] 
        for i in range(self.numOfContours): 
            if self._hierarchy[0,0,i] == -1: 
                if self._perimeters[i] > 10:
                    self.lv1Cnt.append(self._contours[i]) 


def main():
    import pickle 
    # with open("data.pkl", 'rb') as f: 
    #     dataset = pickle.load(f) 
    # visualize(dataset["4cells"][23],save=True,answer=6)
    print("Hello world") 
    with open("../data/data_a_b_570x900_py3.pkl", "rb") as file: 
        data = pickle.load(file)
    testdata = data[18][2]
    # cv2.imwrite("tmp.png", data[18][2]) 
    mgr = ContourManager(testdata) 
    ContourManager.showContours(mgr.getLv1Contours()) 























class FeatureMatcher(object): 

    @staticmethod
    def SIFTdetectorAndBFmatcher(img1, img2):  # correct answer 8
        # Initiate SIFT detector 
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        return len(good) 

    @staticmethod
    def SIFTdetectorAndFLANNmatcher(img1, img2): 
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=10)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=1)
        total_distance = 0 

        # Need to draw only good matches, so create a mask
        # matchesMask = [[0,0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        for i,(m,) in enumerate(matches):
            # if m.distance < 0.7*n.distance:
                # matchesMask[i]=[1,0]
            total_distance += m.distance
        return total_distance
        # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.imshow(img3),plt.show()
    
    @staticmethod
    def ORBdetectorAndBFmatcher(img1, img2): 
        orb = cv2.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        return sum([matches[i].distance for i in range(10)])











if __name__ == "__main__":
    main() 


