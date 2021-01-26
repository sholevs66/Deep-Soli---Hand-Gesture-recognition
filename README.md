# Deep-Soli---Hand-Gesture-recognition
This is a Pytorch implementation of Deep Soli paper: 

"S. Wang, J. Song, J. Lien, I. Poupyrev and O. Hilliges, "Interacting with soli: Exploring fine-grained dynamic gesture recognition in the radio-frequency spectrum", Proc. 29th ACM Annu. Symp. User Interface Softw. Technol., pp. 851-860, 2016."

The original authors github is a bit complicated and uses many different packeges. This project is a simple implementation which allows parsing the radar data and training a CNN-LSTM neural network for hand gesture classification. 


## Data loading and preprocessing
Data and train/validation split configuration was taken from paper's author github: https://github.com/simonwsw/deep-soli
After downloading the data, extract is (as defualt) to \dsp folder.
A preprocessing script **'data_pp_tools.load_data()'** loads and parse the data. The data is splite roughly 50/50 to train/validation by the same configuration file from the original authors. Each data point is a sequence of variable length of 1024 dimensional vector which are reshaped to a [32,32] 2d Doppler map [range, velocity]. 
![Alt text](2d_sample.png?raw=true "Title")

The preprocessing script truncates the data points to equal length of 40 (as the authors mention in their paper). 33 & 23 sequences are droped from the training & validation sets respectivly as they have less than 40 samples.

## Parameters
Basic parameters can bee seen at the bottom of **'train.py'** packed by parser.

## Training
**'train.py'** will load and process the data as described above, and train a CNN-LSTM model.
Predictions are calculated based only on the last hidden LSTM state. Training is done using Negative log loss using true labels. Run command for training with default configurations:

`python train.py`

## Logging
For each run, a folder named by date and time will be created with a log file containing info about the parameters. Losses arrays, accuracy arrays and figures are saved there as well.

![Alt text](acc_epochs.png?raw=true "Title")
