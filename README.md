
For your initial work on writing a program that will solve SPM problems, you can use the following, in our Box folder:  aivaslab -> projects -> ravens:

1. SPM_padded are the plain images, one per problem. 

2. SPM_padded_lined are the images with subimages "drawn" on them.  You can see that sometimes the subimages leave out some information, but it's enough to solve the problems, I believe.

3. input_image_coordinates gives the coordinates of the subimages for all 60 problems.  Each row is one "subimage" from the problem, with the first two numbers representing the x-y coordinates of the top-left corner of a single box, and then the second two numbers being the width and height.  There are 10 rows for each 2x2 problem (4 matrix boxes and 6 answer boxes), and 17 rows for each 3x3 problem (9 matrix boxes and 8 answer boxes).  The different problems are separated by a line break.
```
450	120 [two tabs]  233	187
744	120		233	187
450	377		233	187
744	377		233	187
99	862		233	187
541	866		233	187
974	868		233	187
94	1148		233	187
532	1145		233	187
970	1147		233	187
```
So you should be able to use the SPM_padded images plus the input_image_coordinates to easily read in the subimages for each problem.  You can use the SPM_padded_lined images just for your own reference.

Let me know if any questions,

-MK






`data.pkl` 

```python
import pickle

with open("data.pkl","wb") as f: 
    pickle.dump(dataset,f) 
    
with open("data.pkl", 'rb') as f: 
    dataset2 = pickle.load(f) 
```

```python
datastructure:

dataset = {"4cells":[
							[the first image a1
								[sub img1],
								[],
								...
								[last sub img]
							],
							[],
							[]
							...
						], 
			"9cells":[
							[],
							[],
							[],
							...
						]
			}
```







http://www.instructionaldesign.org/theories/gestalt/ 








similarity metrics 
Hausdorff distance https://en.wikipedia.org/wiki/Hausdorff_distance 
>two sets are close in the Hausdorff distance if every point of either set is close to some point of the other set.

install opencv main and extra module package: (need extra model to measure shape similarity which is hausdorff distance)
https://pypi.org/project/opencv-contrib-python/ 
if you already installed opencv main package, use conda create to avoid potential conflicts

[my personal note: use `conda activate gnn` to activate this environment]
[getting this error https://stackoverflow.com/questions/52305578/sift-cv2-xfeatures2d-sift-create-not-working-even-though-have-contrib-instal , changed environment]

```
python=3.7.3
pip install opencv-contrib-python==3.4.2.16 
```  
# problem with opencv version later than 3.4 is that sift alg is pattern and is not included in the package any more 

finally gets `sift = cv2.xfeatures2d.SIFT_create()` work






