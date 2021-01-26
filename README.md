# Deep-Soli---Hand-Gesture-recognition
Implementation of Deep Soli paper: 

"S. Wang, J. Song, J. Lien, I. Poupyrev and O. Hilliges, "Interacting with soli: Exploring fine-grained dynamic gesture recognition in the radio-frequency spectrum", Proc. 29th ACM Annu. Symp. User Interface Softw. Technol., pp. 851-860, 2016."



## Data loading and preprocessing
Data and train/validation split configuration was taken from paper's author github: https://github.com/simonwsw/deep-soli
After downloading the data, extract is (as defualt) to \dsp
A preprocessing script 'data_pp_tools.load_data()' loads and parse the data. The data is splite roughly 50/50 to train/validation by the same configuration file from the original authors. Each data point is a sequence of variable length of a 2d doppler map [range, velocity]. The preprocessing script truncates the data points to equal length of 40 (as the authors mention in their paper). 33 & 23 sequences are droped from the training & validation sets respectivly as they have less than 40 samples.

