# Aerial-Base-Station-ABS---Assisted-5G-Networks-Through-Human-Mobility-Prediction

## Description
Human mobility machine learning refers to the utilization of deep learning techniques for analyzing and modeling human mobility patterns by utilizing large-scale
mobility datasets and neural network architectures. The objective of deep move is
to identify fundamental patterns and predict future movements of individuals and
groups.
Human mobility prediction through deep learning has the potential to benefit society in many ways, such as improving transportation efficiency, emergency
response planning, and public health measures. However, ethical considerations
such as privacy protection, bias mitigation, and transparency must be thoroughly
addressed to ensure responsible and fair development and deployment of deep
learning models.

## Contribution
Write and modified Code for the model - Encoder and Decoder.

## DataSet
1. Four Square Dataset </br>
Link : [Four_Square](https://www.kaggle.com/datasets/chetanism/foursquare-nyc-and-tokyo-checkin-dataset) </br>

2. The GeoLiofe Dataset </br>
Link : [The_GeoLife_Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=52367) </br>

## Accuracy Acheived

We have achieved accuracy of 0.1515624 , which is greater than previous acheived . </br>
### Fig 1 : Code Run1

<img src="https://github.com/manish5200/Aerial-Base-Station-ABS---Assisted-5G-Networks-Through-Human-Mobility-Prediction/blob/main/Pretain_ScreenShots/T1.png" width="500" height="500" />  </br>
</br>

### Fig 2 : Code Run2
<img src="https://github.com/manish5200/Aerial-Base-Station-ABS---Assisted-5G-Networks-Through-Human-Mobility-Prediction/blob/main/Pretain_ScreenShots/T2.png" width="500" height="500" />  </br>

### Fig 3 : Accuracy V/s different models
<img src="https://github.com/manish5200/Aerial-Base-Station-ABS---Assisted-5G-Networks-Through-Human-Mobility-Prediction/blob/main/Pretain_ScreenShots/results.png" width="500" height="500" /> </br>

### Fig 4 : Accuracy 
<img src="https://github.com/manish5200/Aerial-Base-Station-ABS---Assisted-5G-Networks-Through-Human-Mobility-Prediction/blob/main/Pretain_ScreenShots/Accuracy.jpg" width="2500" height="500" /> </br>

## Instructor
Dr. Shailendra Shukla (MNNIT Allahabad)

## Experimental Setup and Results Analysis

1. System Requirements
The project can be easily run on consumer-grade hardware. Any modern computer
with 4 GB of RAM and a 2 GHz or higher processor, should have no trouble
handling the same.
2. Software Requirements </br>
• python 3.8.2  </br>
• numpy 1.18.1  </br>
• torch 1.4.0  </br>

VS Code installation : </br>
• Download the visual studio Code installer for specific system from the official
VS code site. </br>
• Run the installer(VSCodeUserSetup-version), install in default setting.  </br>
• By default for windows, VS code is installed under C: Microsoft VS Code.  </br>
Python:</br>
• Go to the official Python download page for Windows. </br>
• Downloading the Python Installer. </br>
• After the installer is downloaded, execute the .exe file in default setting. </br>
• Add python to the environment variable(opt).  </br>
Pytorch : </br>
• Visit the official Pytorch website, there choose system specific requirements
and run the command generated in your terminal. </br>
• Eg : “pip3 install torch torchvision torchaudio –index-url </br>
• Link : [Download_PyTorch](https://pytorch.org/) </br>

Running the project : </br>
• Open terminal at the project location. </br>
• ”python train.py –data name=foursquare –data path=../data/”  </br>
• Change the command according to the need of arguments. </br>

## Usage
```sh
### Shell Script
chmod +x run/m.sh  # Ensure execute permission
./run/m.sh         # Run the script

