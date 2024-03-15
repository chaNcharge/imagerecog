# Image Classification in PyTorch

## Running
1. Install the required dependencies `pip install -r requirements.txt` **Highly recommended:** Install all dependencies in a Python virtual environment, such as venv
2. Run `python3 imagerecogtrain.py` first (`python3 mpsimagerecogtrain.py` for Metal-enabled GPUs). Make sure you have a CUDA-enabled GPU active.
3. Run `python3 imagerecogtest.py <path/to/image.jpg>` with any JPG image. I have provided a few images randomly gathered from the internet that I used to test accuracy on my own images. I'm not sure if this works with other image formats, only JPG has been tested. 