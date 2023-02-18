"""This file contains the NN-based distance predictor.

Here, you will design the NN module for distance prediction
"""


'''
Dependenceies : 
    PyTorch 
        OpenCV
        Pandas
        IPython
        seaborn


'''

from pathlib import Path

import torch
import cv2
import urllib.request
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors

# NOTE: Very important that the class name remains the same
class Predictor:
    def __init__(self, model_file: Path):
        # TODO: You can use this path to load your trained model.
        self.model_file = model_file


        # Model_detection
        # self.model_detect = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        # Model_depth
        url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
        urllib.request.urlretrieve(url, filename)

        # self.model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        self.model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        # self.model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)


        self.device = torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = None
        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        

    def predict(self, obs) -> float:
        """This is the main predict step of the NN.

        Here, the provided observation is an Image. Your goal is to train a NN that can
        use this image to predict distance to the lead car.

        """

        # Do your magic...
        
        # Inference
        # results_detect = self.model_detect(obs)

        # Results
        # results_detect.print()
        # results_detect.show() #save()  # or .show()

        # results_detect.xyxy[0]  # img1 predictions (tensor)
        # results_detect.pandas().xyxy[0]  # img1 predictions (pandas)


        # 
        

        input_batch = self.transform(obs).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=obs.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
        # np.savetxt("foo.csv", output, delimiter=",")
        # cv2.waitKey(10000)
        # my_data = np.genfromtxt('foo.csv', delimiter=',')

        
        # my_data = my_data[:][300:420]  # crop
        # my_data = np.transpose(my_data)
        # my_data = my_data[:][295:305]  # crop
        # my_data = np.transpose(my_data)

        my_data = output[:][300:420]  # crop
        my_data = np.transpose(my_data)
        my_data = my_data[:][185:195]  # crop
        my_data = np.transpose(my_data)


        # with plt.ion():
        output = output[:][300:420]  # crop
        output = np.transpose(output)
        output = output[:][295:305]  # crop
        output = np.transpose(output)


        output -= my_data

        # output[0][0] = 5500.0
        # output[0][-1] = 0.0
        print(np.shape(output), end="\t\t<<< shape\n")
        print(np.amax(output), end="\t\t<<< max\n")
        print(np.amin(output), end="\t\t<<< min\n")
        print(np.sum(output)/np.shape(output)[1] , end="\t\t<<< avg sum\n")
        plt.imsave('test.png',output)

        
        
        # plt.show()
        # print(output.shape)
        # plt.close('all')
        # print(output.min())
        # print(output.max())
        # img = np.array(img)
        
        cv2.imshow("op", cv2.imread("test.png"))
        cv2.imshow("obs", cv2.cvtColor(obs, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)
        return 30.0



#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie