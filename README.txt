READ ME 

Running the code for the various methods: 

Logistic regression: 
Please have the following files in the same directory as the data files, then run LogReg.py:
- LogReg.py

Feedforward Neural Net: 
Please have the following files in the same directory as the data files, then run main3.py:
- main3.py
- data.py 
- network2.py

naive_x_val.py - this is the simple cross validation file (optional)

CNN: 
Please have the following files in the same directory as the data files, then run CNN_Code.py:
- CNN_Code.py

Inception v3: 
First train the inception model using: 
bazel build tensorflow/examples/image_retraining:retrain
bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir ~/DIRECTORY_WITH_TEST_IMAGES_IN_JPG
Once that it done then to produce the predicitons file. please have the following files in the same directory as the data files, then run classify.py:
- classify.py
