import os
ProjectFolder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
directory_name = os.path.dirname
imgFolder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Images"))
imgTrain = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Images", "Train"))

imgTrainFlamingo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Images", "Train", "FLAMINGO"))
#imgTrainFlamingo = imgTrainFlamingo.replace("\\", "/")
imgTrainBarnOwl = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Images", "Train", "BARN OWL"))
#imgTrainBarnOwl = imgTrainFlamingo.replace("\\", "/")