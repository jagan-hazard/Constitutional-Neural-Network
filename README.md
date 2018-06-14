# Image classification using Convolutional Neural Networks(CNN)

  Convolutional Neural Networks are one of the most common architecture used for Image classification and object detection. It is the base for most of the modern state of art models such as VGG 16, VGG 19, Inception, SqueezeNet and so on. Here, we will create a simple six layer CNN model to classify the given images. 
 
  This is the program written for classifcation of elephant and non-elephant images (Basically binary classification).If you want to perform multiclass classification simply use categorical_crossentropy or some other loss functions.
  
1. Training our model:
    
        Use the train.py for training our model.
   
2. Classify the test images:
    
        Use the class.py for testing our model.

3. Visualize the each layer output :
    
    To visualize we do training and visualizing in the single fine. This is just to get the intution of what each layer does and how does the nodes work and which node is reponsible for classification of each layer.
    
        Use visualize.py for visualizing the each layer out.

Notes: 

   * To classify the Images, we must do trainning followed by classification.
   * data folder will look like this.
        
                data/
            train/
                elephant/
                    ele001.jpg
                    ele002.jpg
                    ...
                others/
                    other001.jpg
                    other002.jpg
                    ...
            validation/
                elephant/
                    ele801.jpg
                    ele802.jpg
                    ...
                others/
                    other801.jpg
                    other802.jpg
                    ...
   * Refer Keras documentation url : https://keras.io/  which is well explained how to explore keras framework.
