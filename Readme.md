# Brendan Krull's ME 759 Final Project for the Fall 2017 Semester

##### Please Note: The dataset used was too large to upload to github. The datasets can be found at this link here: 

http://yann.lecun.com/exdb/mnist/

##### I am hosting a .zip file of the text files here on my dropbox: 
https://www.dropbox.com/s/21myh61btvxf5wq/HandwritingData.zip?dl=0

These files currently need to be in or be linked to the directory of the executable. 
They are text files, the first line of the label file is the number of labels in the file, and the first three lines of the image file are the number of instances, and the dimensions of those inputs (60000 samples at 28x28 pixels/sample in the training files). The test files are laid out the same way, but they only have 10000 samples.


### Also included is the converter functionality in a .ipynb file.


##### Currently, the files are hardcoded for convenience, so to call it from the command line, one needs to give it 

finalProject <epochs> <len> <lrate> <inertia> <hiddenNodes>

int epochs: Number of Training Epochs (50-100 is reasonable)
int len: the number of training samples to use (Max is 60000 for the provided dataset)
double lrate: the learning rate for the neural net
double inertia: a value between 0 and 1 which affects the backpropagation's reliance on the learning rate (0 is normal Neural net Backpropagation, 1 is entirely based on the changes in the weights)
int hiddenNodes: the number of nodes in the hidden layer (Max 1024)