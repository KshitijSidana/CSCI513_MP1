# code reuse partial of https://github.com/harshilpatel312/KITTI-distance-estimation
"""This file contains the NN-based distance predictor.

Here, you will design the NN module for distance prediction
"""
from mp1_distance_predictor.inference_distance import infer_dist
from mp1_distance_predictor.detect import detect_cars

from pathlib import Path
from keras.models import load_model
from keras.models import model_from_json

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
        self.detect_model = None
        self.distance_model = None

        # backup 
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

    def initialize(self):

        self.detect_model = load_model('mp1_distance_predictor/model.h5')
        self.distance_model = self.load_inference_model()

    def load_inference_model(self):
        MODEL = 'model@1535470106'
        WEIGHTS = 'model@1535470106'

        # load json and create model
        json_file = open('mp1_distance_predictor/distance_model_weights/{}.json'.format(MODEL), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights("mp1_distance_predictor/distance_model_weights/{}.h5".format(WEIGHTS))
        print("Loaded model from disk")

        # evaluate loaded model on test data
        loaded_model.compile(loss='mean_squared_error', optimizer='adam')
        return loaded_model

    
    def call_for_backup (self, obs, image):
        input_batch = self.transform(image).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
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

        # my_data = output[:][300:420]  # crop
        # my_data = np.transpose(my_data)
        # my_data = my_data[:][185:195]  # crop
        # my_data = np.transpose(my_data)

        # output = torch.transforms.functional.gaussian_blur(output,[5,5])
        
        # with plt.ion():
        output = output[:][300:420]  # crop
        output = np.transpose(output)
        output = output[:][285:315]  # crop
        output = np.transpose(output)


        # output -= my_data
        avg_sum = np.sum(output)/np.shape(output)[1]
        min = np.amin(output)
        max = np.amax(output)
        # output[0][0] = 5500.0
        # output[0][-1] = 0.0
        print(np.shape(output), end=", shape, ")
        print(max, end=", max, ")
        print(min, end=", min, ")
        print(avg_sum , end=", avg sum, ")
        print(obs.distance_to_lead, end=", d2l, ")
        plt.imsave('test.png',output)

        
        
        # plt.show()
        # print(output.shape)
        # plt.close('all')
        # print(output.min())
        # print(output.max())
        # img = np.array(img)
        
        cv2.imshow("op", cv2.imread("test.png"))
        cv2.imshow("obs", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

        F = min
        D = max
        N = (((1/(abs(F/D))**0.5)*10)**0.5) *3
        R = F/5

        ret = ((1/((avg_sum-40000)**0.5)*1000) + (N-20) ) - 0.5*(10-R)
        # ret =((1/((avg_sum-40000)**0.5))*1000 + (N14-20) ) - 0.5*(10-min)        
        # ret = ((1/ret)**0.5)*2000
        print(ret, end=", ret\n")
        return ret
        

    def predict(self, obs, image) -> float:
        """This is the main predict step of the NN.

        Here, the provided observation is an Image. Your goal is to train a NN that can
        use this image to predict distance to the lead car.

        """
        data = obs
        image_name = 'camera_images/vision_input.png'
        # load a trained yolov3 model

        car_bounding_box = detect_cars(self.detect_model, image_name)  # return the bounding box of car
        dist_test = data.distance_to_lead
        # Different dist_test will have effect on the prediction
        # You can play with the number of dist_test
        if car_bounding_box is not None:
            dist = infer_dist(self.distance_model, car_bounding_box, [[dist_test]])
        else:
            print("No car detected! Calling Backup!")
            # If no car detected what would you do for the distance prediction
            # Do your magic...
 
            dist = self.call_for_backup(obs, image)
            if dist > 15:
                dist = 30

        # print("estimated distance: ", dist)



        return dist
