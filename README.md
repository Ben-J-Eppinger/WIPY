# WIPY

Waveform Inversion in PYthon

If you use this software, please cite: 
Eppinger, B. J., Holbrook, W. S., Liu, Z.,Flinchum, B. A., & Tromp, J. (2024). 2d near‐surface full‐waveform tomography reveals bedrock controls on critical zone architecture. Earth and Space Science, 11, e2023EA003248. https://doi.org/10.1029/2023EA003248

Usage Instrctions

Set up a new anaconda environment 
List  your current conda environments: 
 conda info --envs 
 
Create a new environment for WIPY with python installed. 
  conda create -n WIPY python 

You can use conda list to see the default packages installed, such as pip. 
*Note that in order to use the WIPY environment in a Jupyter notebook, you may need to install additional packages. 
Activate the WIPY environment with 
  conda activate WIPY 
and use 
  pip install <package_name> 
to install all the following packages:  
Numpy 
Matplotlib 
Obspy 
Scipy  
joblib 

If you need to delete your conda environment and try again, you can use the following command: 
  conda remove --name ENV_NAME –all 
 
Installing WIPY 
Download WIPY from GitHub: https://github.com/Ben-J-Eppinger/WIPY  
Navigate to the WIPY folder ([your directory]/WIPY/) 
The WIPY directory should contain the following files:
Examples 
README.md 
setup.py 
wipy 
 
Now you can install WIPY using: 
  pip install -e . 
The last step is to add WIPY to your $PATH variable.  Edit your $PATH variable in your .bashrc  (.zshrc if on a Mac) file. Add the line: 
  PATH=/path/to/WIPY/wipy/work_flows:$PATH

Save your edits, then run 
  source .bashrc (or source .zshrc) to apply the changes. 
  
This will allow you to run the script wipy_run in the terminal from any directory on your system (you may have to give wipy_run execution permission using chmod +x). 
