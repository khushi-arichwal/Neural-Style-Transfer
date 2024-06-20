# Neural Style Transfer

This project is an implementation of Neural Style Transfer using TensorFlow and Streamlit. 
## Files

1. `nst.py`: This file contains the core implementation of the Neural Style Transfer algorithm. It includes functions for loading images, creating a mini model, calculating the gram matrix, defining the custom style model, calculating the total loss, and performing a training step.

2. `webpage.py`: This file uses Streamlit to create a web interface for the Neural Style Transfer application. It allows users to upload their own content and style images or select from a predefined list. The processed image is then displayed on the webpage. 

## How to Run

1. Install the required packages: `pip install -r requirements.txt`
2. Run the Streamlit app: `streamlit run webpage.py`

## How it Works
Neural Style Transfer is a technique that applies the style of one image to the content of another image using Convolutional Neural Networks. The algorithm works by minimizing the difference between the content and style of the input images. 

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
