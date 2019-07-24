# Generate Expressionist Art
Deep convolutional generative adversarial network (DCGAN) to generate pictures of expressionist art. This implementation is done in Keras with tensorflow backend and is written in Python. <br><br>
This was trained on ~800 images of expressionist art scraped from Google images and processed with openCV.<br><br>
I didn't have access to a GPU so I ran the algorithm on AWS using an EC2 C5 instance for around a day, and produced an image every epoch to see the improvement.  <br><br>
A big challenge I came across with this project was that my losses for both discriminative and generative networks quickly went to 0 after some number of epochs, hence causing any training to cease since the gradient would also go to 0. After trying to fiddle around with the network for a little bit to try to fix this including number of layers, numbers of filters, filter sizes, etc., I tried adding artificial "noise" to the dataset by not using hard labels "0" or "1" for generated vs real images; I used a random number between 0-0.1 and 0.9-1 instead. This fixed my earlier problem.

## Use:
To use, run: `python3 lenet_driver.py [options]`  
>Options:  
    -s, --save-model    Flag to train and save a model  
    -l, --load-model    Flag to load an already trained model  
    -t, --train-model   Flag to load and train an already trained model (to continue training)
    -w, --weights       Filename of the weights file for the already trained model. (Must be in output directory and hdf5 extension)  
