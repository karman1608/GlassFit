## GlassFit

### Problem Definition:
The goal of this project is to develop a system that can automatically classify human faces into 5 different shape categories, such as round, oval, square, oblong or heart-shaped. The system will take as input a collection of facial images in .jpeg format, and use deep learning techniques to extract and analyze facial features, allowing it to accurately classify each image according to its corresponding face shape category. The project will involve building a CNN-based model to train and test the face shape classification system, as well as implementing various performance evaluation metrics to assess the effectiveness of the model. The final system will then recommend spectacles frame according to the face shape. For example, square frame looks good for round face shape.

### Study and Understanding of Algorithm
We would use a CNN Model for Detecting face shape Here is an overview of the algorithm for using a CNN on preprocessed images for face shape detection:
1. First, you need to preprocess the input images to make them suitable for feeding into the CNN. This can involve steps such as resizing the images to a uniform size, normalizing the pixel values, and converting the images to grayscale.
2. Next, you need to split the preprocessed images into training, validation, and test sets. The training set is used to train the CNN, the validation set is used to evaluate the performance of the CNN during training and make adjustments to the model if necessary, and the test set is used to evaluate the final performance of the CNN.
3. Define the architecture of the CNN. This involves specifying the number and type of layers in the network, the activation functions used in each layer, and the connections between layers.
4. Train the CNN using the preprocessed training set. During training, the CNN learns to recognize patterns in the images that correspond to different face shapes.
5. Evaluate the performance of the CNN using the validation set. This can involve measuring metrics such as accuracy, precision, recall, and F1 score.
6. Fine-tune the model if necessary by adjusting the hyperparameters of the CNN, such as the learning rate and the number of epochs.
7. Test the final performance of the CNN using the preprocessed test set. This can involve measuring the same performance metrics as in step 5.
8. Use the trained CNN to classify new images of faces according to their shape. This involves feeding the preprocessed image into the CNN and obtaining the predicted shape label from the output layer.
<p>Note that there can be many variations and nuances to the above algorithm depending on the specific requirements of the face shape detection task and the nature of the preprocessed images.</p>
<p>The number of layers used in a CNN for face shape detection can vary depending on the specific require- ments of the task and the complexity of the images. However, typically a CNN used for image classification tasks such as face shape detection would consist of multiple layers, including convolutional layers, pooling layers, and fully connected layers.</p>
<p>A typical CNN architecture for image classification may have several convolutional layers, followed by pooling layers to downsample the feature maps and reduce their spatial dimensions. The output of the last pooling layer is then flattened and fed into one or more fully connected layers to produce the final output.</p>
<p>For face shape detection specifically, it is common to use a pre-trained model as a starting point and fine-tune it on the specific task at hand. We would use a pre-trained model such as <b>VGG16 </b>, which have many layers and have been trained on large datasets of natural images, and then adapt the model for face shape detection by modifying the output layer and fine-tuning the weights on a dataset of preprocessed face images.</p>


## How to use this repository

Clone the this repository by <code>git clone https://github.com/nilayp2107/GlassFit.git</code>
<br /><br />Now create a virtual python environment 
<br /><code>\#Create a virtual environment (optional)
python -m venv myenv</code>
<code>
\# Activate the virtual environment
source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
</code>

<br><br> Install all the modules in the environment like tensorflow,mtcnn,pandas,numpy.

<br> Finally run the app by <code>python3 app.py</code>

## Snapshots of the website
### Home Page
<img width="1723" alt="Screenshot 2023-09-06 at 10 14 33 PM" src="https://github.com/nilayp2107/GlassFit/assets/75634739/9d7ac012-a9a5-4fbc-956f-6ef619a53d15">

### Prediction Page
<img width="1727" alt="Screenshot 2023-09-06 at 10 15 05 PM" src="https://github.com/nilayp2107/GlassFit/assets/75634739/54b59de3-71ed-45b2-9e98-b86c9ee38d5c">
