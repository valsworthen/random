This tutorial aims to configure a Deep Learning AMI on AWS from a basic Ubuntu installation.  

I may try to adapt it for an installation on a __local__ ubuntu distribution. In particular this second case requires to shut down lightdm during the CUDA installation part! [(see here until step 14)](https://kislayabhi.github.io/Installing_CUDA_with_Ubuntu/)  

##### Table of Contents  
[Basic AMI Configuration](#basic-ami-configuration)  
[Deep Learning tools installation](#deep-learning-tools-installation)  

# Basic AMI configuration

Select the **Ubuntu Server 16.04 LTS (HVM), SSD Volume Type - ami-df8406b0** AMI

## General packages configuration
```bash
sudo apt update  

sudo apt install python3-pip  
pip3 install --upgrade pip  

pip3 install pandas --user  
pip3 install scipy --user  
pip3 install scikit-image --user  
```
## Configure Jupyter

First install Jupyter for python3  
```bash
pip3 install jupyter --user  
```
We need to configure the jupyter server. You can follow the instruction from the aws doc: http://docs.aws.amazon.com/mxnet/latest/dg/setup-jupyter-configure-server.html  

In a nuthsell:
```bash
cd
mkdir ssl
cd ssl
sudo openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout "cert.key" -out "cert.pem" -batch
```
Set password to access the notebooks. Open ipython and type:
```bash
from IPython.lib import passwd
passwd()
```

Enter you password when prompted, then copy and paste the hashed password and exit the ipython console.  
After that we create the config file for jupyter:   
```bash
jupyter notebook --generate-config
vi ~/.jupyter/jupyter_notebook_config.py
```

Append the following lines at the end of the file:
```bash
c = get_config()  # Get the config object.
c.NotebookApp.certfile = u'/home/ubuntu/ssl/cert.pem' # path to the certificate we generated
c.NotebookApp.keyfile = u'/home/ubuntu/ssl/cert.key' # path to the certificate key we generated
c.IPKernelApp.pylab = 'inline'  # in-line figure when using Matplotlib
c.NotebookApp.ip = '*'  # Serve notebooks locally.
c.NotebookApp.open_browser = False  # Do not open a browser window by default when using notebooks.
c.NotebookApp.password = #Hashed password  
```
You can then launch jupyter notebook and access it with the public ip of the AWS instance. **Make sure to use HTTPS** and confirm when security issue appears.  
```bash
jupyter notebook
```
https://PUBLIC_IP:8888 then confirm security

***

# Deep Learning tools installation

## Install Cuda  

Test if gpu is detected:  
```bash
lspci -nnk | grep -i nvidia  
```
Install some mystic libraries:
```bash
sudo apt-get update  
sudo apt-get install libglu1-mesa libxi-dev libxmu-dev -y  
```
### Install the nvidia driver for your gpu.
You may find the link for the correct driver on [the Nvidia website](http://www.nvidia.fr/Download/index.aspx).  
```bash
wget LINK/TO/DRIVER  #should be a .run file

sudo chmod +x NVIDIA-/DRIVER/FILE.run  
./NVIDIA-/DRIVER/FILE.run  
```
Test if driver is correctly installed:  
```bash
nvidia-smi  
```
### Download and install Cuda:
**Please note that Tensorflow 1.4 IS NOT compatible with CUDA 9**   
CUDA is a large file to download (~1.5Gb)  
You can find the download link for the appropriate version [here (link to CUDA 8)](https://developer.nvidia.com/cuda-80-ga2-download-archive)  
```bash
wget LINK/TO/CUDA.RUN   #should be a .run file  
```  

Make sure **NOT TO INSTALL THE DRIVER FROM CUDA** since you already installed a newer version earlier.  

```bash
sudo chmod +x cuda_VERSION_linux.run  
./cuda_VERSION_linux.run  #maybe you'll need to add --override
```
Modify bashrc to add CUDA libraries to the PATH:  
```bash
echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"' >> ~/.bashrc  
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc  
echo 'export PATH="/usr/local/cuda-8.0/bin:$PATH"' >> ~/.bashrc  
```
Test if CUDA installation worked:   
You may need to reboot or to source .bashrc  
```bash
nvcc --version  
```
## Install cuDNN  
cuDNN is a Nvidia library for deep learning  
Subscription to the nvidia developer program is mandatory to download cuDNN. If you really need to download from command line, you can use the following wget line. Otherwise, just download the .tgz file from [cuDNN website](https://developer.nvidia.com/rdp/cudnn-download)  
```bash
wget http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/cudnn-8.0-linux-x64-v6.0.tgz  
tar -xzvf cudnn-6.0-linux-x64-v4.0-prod.tgz  
```
The we need to move the libraries into the CUDA installation folder.  
```bash
cp cuda/lib64/* /usr/local/cuda/lib64/  
cp cuda/include/cudnn.h /usr/local/cuda/include/  
```
## Install Tensorflow

This should be the easy part of this installation. The following steps are clearly explained on the Tensorflow website.  

Install library (just to be sure)  
```bash
sudo apt-get install libcupti-dev  
```
If you want to use virtual environments, use the next three lines. Otherwise just skip to the pip install:  
```bash
sudo apt-get install python3-pip python3-dev python-virtualenv  
virtualenv --system-site-packages -p python3 **tensorflow**  
source ~/tensorflow/bin/activate  
```
```bash
pip3 install --upgrade tensorflow-gpu --user
```
Then test the installation with the following python script:  
```bash
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
# Start tf session
sess = tf.Session()
# Run the op
print(sess.run(hello))
```
## (Optional) Install Keras  
```bash
pip3 install keras --user
```
Test in python script:  
```bash
import keras
```
