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
        search_params = dict(checks=50)   # or pass empty dictionary
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


def main():
    import pickle 
    with open("data.pkl", 'rb') as f: 
        dataset = pickle.load(f) 
    # visualize(dataset["4cells"][23])
    visualize(dataset["4cells"][23],save=True,answer=6)
    # for i in range(23):
    #     visualize(dataset["4cells"][i])










if __name__ == "__main__":
    main() 


