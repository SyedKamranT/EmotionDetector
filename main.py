
import matplotlib.pyplot as plt # library matplot for imread and plt as variable
from deepface import DeepFace #deepface library for the detection which includes the libraries - keras,numpy,etc
from matplotlib.image import imread #importing imread to read the image in numbers or (str)

#notify 'please close the image after view'
img1 = imread('images/download (9).jpg') #import the image address at  ()
plt.imshow(img1[:, :, ::-1]) #displaying the image with placement values
plt.show()

result = DeepFace.analyze(img1, actions=['emotion']) #analyzing the emotions using deepface library.

print(result) #displaying the image's dominant value


#!---please close the image after the view for the results---!#
