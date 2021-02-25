# Image_colorization

```
Original Image                                              Output of Regression Network
```

<div align='center'>
    <img src='test/t3.jpg' height="256" width='50%'>
    <img src='test/colorized_reg_t3.jpg' height="256" width='50%'>
</div>

```
                                      Output of U-Net based Network
```

<div align='center'>
    <img src='test/colorized_unet_t3.jpg' height="512" width='100%'>
</div>



## Files

```
├── notebooks                   
|    ├── Image_colorizer.ipynb        # Jupyter Notebook with trained model used to generate output
|    ├── Image_Preprocessing.ipynb    # Jupyter Notebook for preprocessing of dataset images
|    ├── Train and Results.ipynb      # Jupyter Notebook for training of the models
|    └── model.ipynb                  # Jupyter Notebook contatining model architecture.
|                                           
├── reg_imcolor
|    |
|    ├── reg_imcolor.h5               # model weights of trained regression network 
|    └── reg_imcolor.json             # model architecture of regression network 
|
├── unet_imcolor
|    |
|    ├── unet_imcolor.h5              # model weights of trained U-Net model 
|    └── unet_imcolor.json            # model architecture of U-Net model 
|
├── test                              # Contains input and output images
|
├── Image_colorizer.py                # Python file with trained model used to generate output  
|
├── model.py                          #Python file contatining model architecture.
|
└── README.md

```

### STEPS TO RUN IN LOCAL SYSTEM-

```
1. Download the repository in your local system
2. Run "python -m ./Image_colorizer.py model filename" (where model=0 for regression network and model=1 for unet model)
```
