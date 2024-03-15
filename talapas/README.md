# Image Classification in PyTorch
Modules:
pytorch

## Running
1. Load the pytorch module using `module load pytorch`
2. Run `python3 imagerecogtrain.py` or use the provided sbatch script `train` first. Make sure you have a CUDA-enabled GPU active. 
3. Run `python3 imagerecogtest.py <path/to/image.jpg>` with any JPG image or use the provided sbatch script `test`. I have provided a few images randomly gathered from the internet that I used to test accuracy on my own images. I'm not sure if this works with other image formats, only JPG has been tested. 