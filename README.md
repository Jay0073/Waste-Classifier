# Week-1
# **CNN model for Waste Mangement**

## Project Description
This project aims to design and train a deep learning model that can accurately distinguish between organic and recyclable waste images. The model can be integrated into waste management systems to automate the sorting process, reducing contamination rates and increasing recycling efficiency.

## Featuers
 - Convolutional Neural Network (CNN) architecture
 - Image Preprocessing and Data Augmentation
 - Training and Validation on a dataset of organic and recyclable images
 - Model evaluation using metrics such as accuracy, precision, and recall

## Technologies Used
- Python
- TensorFlow/Keras
- OpenCV
- NumPy
- Matplotlib

## Dataset
- URL: https://www.kaggle.com/datasets/techsash/waste-classification-data/data
- Dataset size: 211mb consisting of 25, 077 images
- Classes: Organic, Recyclable
- Class balance: 56 : 44

## How to Use
 - To get started, clone the repository using the following command:
   ```bash
   git clone https://github.com/Jay0073/Waste-Classifier.git
   cd Waste-Classifier
 - Ensure you have python installed, then install the required libraries:
   ```bash
   pip install -r requirements.txt
 - after installing the libraries, you can run the streamlit app to check the model performance
   ```bash
   python -m streamlit run app.py
 - Alternatively, if you want to see the model training and stuff then checkout **CNN.ipynb** file
   
## See the Deployed Model
You can see the deployed model in action by visiting the following link: https://waste-classifier-busdsh36gmgochhha3ovz6.streamlit.app/

## Contributions
Contributions are welcome! If you would like to contribute, please follow these steps:
 - Fork this repository
 - Create a new branch
   ```bash
   git checkout -b branch-name
 - Make your changes and commit them
   ```bash
   git commit -m 'commit-message'
 - Push to the branch
   ```bash
   git push origin branch-name
 - Open a pull request
