


**Methodology**
1. Data Acquisition and Preparation
- Dataset Access: The dataset is acquired from Kaggle (https://https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection), containing MRI images with brain tumors
   ("yes" directory for positive samples) and without ("no" directory for negative samples)
- File Handling: The dataset is downloaded and extracted using Python's zipfile module,
   ensuring that all the necessary image files are accessible for processing.
  - Directories: directories to both subfolders are defined
2. Data Pre-processing
- Image Loading and Transformation: Images are loaded and resized to a uniform size (128x128
     pixels) using OpenCV to standardize the input for the neural network. They are also
     converted from BGR to RGB color space to match the expected input format for color images
     in most deep learning frameworks.
- Data Normalization: Pixel values are normalized to the range [0,1] by dividing by 255.0,
   facilitating more stable training dynamics.
3. Model Design and Implementation
- CNN Architecture: The model comprises multiple convolutional layers with pooling layers to extract features from the images, followed by fully connected layers to perform classification based on these features.
- Activation Functions: Non-linear activation functions like Tanh are used to introduce non-linearities into the model, helping it to learn more complex patterns.
- Output Layer: The final layer uses a sigmoid activation function to provide a probability score indicating the presence of a tumor.
4. Training the Model
Loss Function: Binary Cross-Entropy Loss is used as it is suitable for binary classification problems.
- Optimizer: An Adam optimizer is used for adjusting the weights of the network during training, with a learning rate specified to control the step size of the weight updates.
- Batch Processing: Data is loaded in batches using PyTorch’s DataLoader to make the training process efficient and manageable.
- Training Loop: The model is trained over multiple epochs, updating the weights based on the loss calculated for each batch.
5. Model Evaluation
- Performance Metrics: After training, the model’s performance is evaluated using metrics like accuracy, and the results are visualized through a confusion matrix.
- Thresholding: Output probabilities are thresholded to classify each image as having a tumor or not, based on a chosen cutoff value (e.g., 0.5).
6. Visualization and Analysis
- Sample Outputs: Sample images and their labels are plotted to provide visual confirmation of the model’s predictive capabilities.
- Error Analysis: By reviewing the types and frequencies of errors (false positives and false negatives) in the confusion matrix, insights into the model's strengths and weaknesses are gained.
7. Adjustments and Iterative Improvement
- Hyperparameter Tuning: Parameters like learning rate and number of epochs are adjusted based on the model's performance to find the optimal settings.
- Model Architecture Tuning: Modifications to the CNN architecture (e.g., adding layers, changing layer sizes, or pooling strategies) are considered to improve accuracy.
8. Deployment Preparation
  - Model Saving and Loading: For operational use, the trained model can be saved and later
    loaded to make predictions on new MRI images.
