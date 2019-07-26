# Generate Expressionist Art
Deep convolutional generative adversarial network ([DCGAN](https://arxiv.org/pdf/1511.06434.pdf)) to generate pictures of expressionist art. This implementation is done in Keras with tensorflow backend and is written in Python. <br><br>
This was trained on ~2000 images of expressionist art scraped from Google images and processed with openCV.<br><br>
I didn't have access to a GPU so I ran the algorithm on AWS using an EC2 C5 instance for just about 24 hours, and produced an image every epoch to see the improvement.  <br><br>

## Use:
To use, run: `python3 lenet_driver.py [options]`  
>Options:  
    -s, --save-model    Flag to train and save a model  
    -l, --load-model    Flag to load an already trained model  
    -t, --train-model   Flag to load and train an already trained model (to continue training)
    -w, --weights       Filename of the weights file for the already trained model. (Must be in output directory and hdf5 extension)  


## Challenges
A big challenge I came across with this project was that my losses for both discriminative and generative networks quickly went to 0 after some number of epochs, hence causing any training to cease since the gradient would also go to 0. I didn't get much progress by trying to fiddle around with the network such as adjusting number of layers, numbers of filters, filter sizes, etc. <br><br>
I tried adding artificial "noise" to the dataset by not using hard labels "0" or "1" for generated vs real images; I used a random number between 0-0.1 and 0.9-1 which solved my problem of vanishing loss. <br><br>
(TO BE CONTINUED)