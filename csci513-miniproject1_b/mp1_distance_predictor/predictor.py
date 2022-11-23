"""This file contains the NN-based distance predictor.

Here, you will design the NN module for distance prediction
"""

from pathlib import Path


# NOTE: Very important that the class name remains the same
class Predictor:
    def __init__(self, model_file: Path):
        # TODO: You can use this path to load your trained model.
        self.model_file = model_file

    def predict(self, obs) -> float:
        """This is the main predict step of the NN.

        Here, the provided observation is an Image. Your goal is to train a NN that can
        use this image to predict distance to the lead car.

        """

        # Do your magic...

        return 30.0
