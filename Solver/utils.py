from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
import datetime,time
import os 



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


