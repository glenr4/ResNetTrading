# ResNetTrading

Transfer learning project using ResNet50 model and retrained on candlestick charts showing either up, down or sideways price action. The aim is to be able to predict direction given an unseen chart.


## Restore
1. Make sure you have Miniconda installed: https://www.anaconda.com/docs/getting-started/miniconda/install
1. Setup the environment: `conda env create -f environment.yml`
1. Activate the environment: `conda activate ResNetTransfer`



# TODO: update items below
## Run
### Train
* Run: `python main.py`

### Predict and Evaluate on Test Data
* Run: `python predict.py`

### Predict and Show a Single Image
* Run: `python predict_single_img.py`

## Enable GPU
Creating the environment from `environment.yml` should mean that this isn't necessary but for reference:
1. `pip install tensorflow==2.10.0` # last version that supports Windows
1. `conda install cudatoolkit=11.2 cudnn=8.1 -c=conda-forge`
1. `pip install --upgrade tensorflow-gpu==2.10.0`

https://stackoverflow.com/a/78351204

### Check GPU
* Run: `python check_GPU.py`
