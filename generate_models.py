import glob
import os
import sys
import ultralytics

"""
Pass in the last running model as the argument for this generator
or the base model that you want to train data over
"""
base_model = sys.argv[1]
model = ultralytics.YOLO(base_model)

"""
Pass in the directory that all the training should be resolved from.
Install all the training data into the directory first before
continuing
"""
base_directory = sys.argv[2]

"""
Pass in the format of the model as a third argument. Use onnx by
default
"""
model_format = sys.argv[3]

yaml_files = glob.glob(os.path.join(base_directory, '**', '*.yaml'), recursive=True) + \
             glob.glob(os.path.join(base_directory, '**', '*.yml'), recursive=True)

for yaml_file in yaml_files:
    model.train(data=yaml_file, epochs=40, imgsz=640)

"""
The path to the generated model. 

This operation should take a couple of hours to many days to actually complete
"""
print(model.export(format=model_format))
