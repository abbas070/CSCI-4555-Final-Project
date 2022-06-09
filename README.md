# Project Background

The purpose of this project is to create a model to classify images into 10 predetermined classes. Throughout our work on the project, we constantly attempted to reduce the loss and improve the accuracy of our model. For this project we decided to use Python for implementation, for several reasons:

- We wanted to gain a better understanding of the language.
- Python is the best language available for working with machine learning models and algorithms; Python has many libraries that support machine learning projects such as tensorflow.
- Python libraries allow you to access, process, and transform your data very effectively.

Our hypotheses was that the model would have a good-to-high accuracy rate, with a somewhat slow run time, which would still be improvable using more advanced techniques.

# Data Description

For this project we used the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) image dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset contains 50,000 training images and 10,000 testing images:

- The dataset is divided into 5 training batches and 1 testing batch, each with 10,000 images. 
- The test batch contains exactly 1,000 randomly-selected images from each class.

The 10 classes for the images to be classified in, which have labels 0-9 are:

- airplane: 0
- automobile: 1 (includes sedans, SUVs, etc...)
- bird: 2
- cat: 3
- deer: 4
- dog: 5
- frog: 6
- horse: 7
- ship: 8
- truck: 9 (includes only big trucks)

Those classes are completely mutually exclusive, where there is no overlap between automobiles and trucks.

## Training Set

Here is a plot of the first 25 images in the training set, with the respective class each is assigned to, displayed beneath each image:

![Training Set Figure](https://user-images.githubusercontent.com/60119746/167994547-49c17d9d-7a1f-4f9a-b7e6-1dec3d8e8cf2.png)

## Testing set

Here is a plot of the first 25 images in the training set, where we need our model to predict which class each image falls under:

![Testing Set Figure](https://user-images.githubusercontent.com/60119746/167994887-e5b17fce-7956-4534-b2d2-5f30f6efb3fb.png)

# Method

The steps taken to implement the model are as follows:

- Download and normalize the CIFAR-10 dataset
- Create a convolutional base
- Introduce Dense layers
- Compile and train the model

## Normalizing the data

```
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0
```

The first step of this project is to download the dataset to be used, which we did through Python. After that we normalize the pixel values of the images to be between 0 and 1, which was done by dividing original pixel values of the images by 255.

## Convolutional Base

Since we are dealing with an image-based dataset, we decided to go with a Convolutional Neural Network (CNNs) to classify the images into their corresponding classes.

```
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
```
For our convolutional base, we decided after multiple tests to go with 3 [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D) layers and 2 [MaxPooling2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D) layers between them. The CNN takes the tensors of the height, width, and number of color channels of the images, which we configured for this CNN to process input 32x32x3 (keep in mind 32x32 is dimensions of the data images). We used ReLU as the activation function for the model in all the layers of the CNN.

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 30, 30, 32)        896

 max_pooling2d (MaxPooling2D  (None, 15, 15, 32)       0
 )

 conv2d_1 (Conv2D)           (None, 13, 13, 64)        18496

 max_pooling2d_1 (MaxPooling  (None, 6, 6, 64)         0
 2D)

 conv2d_2 (Conv2D)           (None, 4, 4, 64)          36928

=================================================================
Total params: 56,320
Trainable params: 56,320
Non-trainable params: 0
```

As can be seen in the model summary above, the width and height dimensions of the layers tend to shrink as you go deeper in the network. The number of output channels for each Conv2D layer is controlled by the first Conv2D, which has the input image configuration in it.

## Dense Layers

```
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='softmax'))
model.add(layers.Dense(10))
```

After implementing the convolutional base, we added two [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) layers on top of the final output layer. It is usually recommended to use two Dense layers rather than just one, so we proceeded with that. The first dense layer is of shape 4x4x64, which is a 3D tensor. The second layer should be a 1D tensor, so we first flattened our output into 1D from 3D. Since our data has 10 output classes, the second (final) Dense layer should also have 10 outputs, and used softmax as its optimization function (more on this in the Process section).


Here is the final model summary:

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 30, 30, 32)        896

 max_pooling2d (MaxPooling2D  (None, 15, 15, 32)       0
 )

 conv2d_1 (Conv2D)           (None, 13, 13, 64)        18496

 max_pooling2d_1 (MaxPooling  (None, 6, 6, 64)         0
 2D)

 conv2d_2 (Conv2D)           (None, 4, 4, 64)          36928

 flatten (Flatten)           (None, 1024)              0

 dense (Dense)               (None, 64)                65600

 dense_1 (Dense)             (None, 10)                650

=================================================================
Total params: 122,570
Trainable params: 122,570
Non-trainable params: 0
```

## Compiling and Training

```
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))
```

We used Adam for the optimization of the model since it is arguably the best among the adaptive optimizers. We also decided that 10 epochs is the most optimal when considering accuracy and the average time it takes to compile and train the model due to the enormous size of the data.

# Process

We tried:

- Different amount of layers
- Different amount of epochs
- Different activation functions in final Dense layer.
- Different dropout rates.

We found:

- The number of layers we used is optimal for the model (in our case); adding more layers to the CNN did not really improve the accuracy or loss of the model, so we decided to stick with what seemed as a reasonable amount of layers.
- 10 epochs is already plenty specially with the size of the dataset we used. On average running the model took around 2-5 minutes each time (depending on the computer used). Given that, adding more epochs would just slow the process down even more than it already is. Also, we tried running the model with 20 and 25 epochs, and the accuracy and loss differences were not worth the additional time it takes to compile and run the model. Using less than 10 epochs did show a significant drop in accuracy and loss. 
- Using 3-4 epochs results in accurate results when compared to the training data, but we figured that 3-4 epochs is not enough so we stuck with 10.
- We tried using both softmax and ReLU activation functions in the final Dense layer but didn't see a major difference in accuracy for the model. At the time of the presentation we thought of using ReLU activation in the final Dense layer, but after checking different sources, we noticed that they all used softmax in their final layer so we switched to that.
- We tried adding a dropout regularization however it did not really improve the accuracy of the model, and did not help much with overfitting.

# Results

![Graph](https://user-images.githubusercontent.com/60119746/168017328-d79878cc-464b-4d86-a973-9556623ef904.png)

We can see that the model is overfitting here, which can be improved.

- Training accuracy: between 0.76 and 0.79 (differs every run)
- Training loss: between 0.62 and 0.65 (differs every run)
- Test accuracy: between 0.71 and 0.73 (differs every run)
- Test loss: between 0.86 and 0.88 (differs every run)

Our hypotheses was somewhat correct, however we did not really think much of the loss or overfitting when we first started working on the model, which we believe can be improved. We also learned that using datasets as large as CIFAR-10 maybe wasn't the best idea, due to not being to fully see and understand all the images provided, also it is really time consuming for the model to go through this many images.

# Conclusion

## Challenges

- Slow run times.
- Understanding how to use Python for the first time, and getting the hang of the packages we used.
- Implementation of CNN required the use of several packages and documentation first.
- Implementation of regularization L1 and/or L2; they dropped the accuracy rate by a lot so we decided to not use them.
- Model is overfitting.

## Opportunities for Improvement

- Speed up runtime of the model.
- Test model with more advanced CNN techniques.
- Improve overfitting by adding proper regularization (which we apparently did a bad job at).
- Reduce loss.
