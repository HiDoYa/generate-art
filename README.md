# Generate Expressionist Art
Deep convolutional generative adversarial network (DCGAN) to generate pictures of expressionist art. This implementation is done in Keras with tensorflow backend and is written in Python. <br><br>
This was trained on ~800 images of expressionist art scraped from Google images and processed with openCV.<br><br>

## Use:
To use, run: `python3 lenet_driver.py [options]`  
>Options:  
    -s, --save-model    Flag to train and save a model  
    -l, --load-model    Flag to load an already trained model  
    -w, --weights       Filename of the weights file for the already trained model. (Must be in output directory and hdf5 extension)  

