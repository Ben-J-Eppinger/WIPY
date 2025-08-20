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

Running the box example 
We recommend that you open VS Code (Visual Studio Code), then “Open Folder” and open the “box” folder under Examples. To open a terminal window within VS Code, type ctrl-~ (“control-tilde”). This will open a terminal window in your current working directory. 

First we need to copy the compiled binary files on your machine into your current working directory. In the terminal window, in the …/Examples/box directory, do the following:
 cp -r ~/specfem2d-master/bin specfem2d/ 
 mkdir specfem2d/OUTPUT_FILES
*Note, you must edit the paths according to the directory srtucture on your own machine. 

Test specfem 2d
Test specfem2d by running 
	% cd specfem2d
	% ./bin/xmeshfem2D
	% ./bin/xspecfem2D
Notice how the model and simulation parameters have changed from the original example. Look at the source, stations, interfaces, and parameters files to understand why.

Generate “observed data” 
Navigate to [your directory]/WIPY/Examples/box/  
Review parameters.py. WIPY reads its parameters from this file.
 
Fill out the paths.py file with your environment as follows: 
 
wipy_root_path = “/path_to/WIPY/Examples/box/” 
solver_exec_path = “/path_to/specfem2d/bin/” 
solver_data_path = “/path_to/WIPY/Examples/box/DATA/” 
#obs_data_path: comment this variable out for now 
model_init_path = “/path_to/WIPY/Examples/box/model_true/” 
 
*There is no need to delete the comments in this file 
*If you run paths.py and nothing  prints out, then the paths you input are valid (that is to say, you did not make any typos). Remember that you have to manually save (cmd-S) the paths.py file every time you edit it for the changes to activate.
 python paths.py 
 
Use forward_test to generate “observed” data from model_true. This creates the OUTPUT directory and files therein. 
 conda  activate WIPY 
 wipy_run forward_test 
 
Transfer the generated synthetic data to a new folder. Next, we’ll invert this data. 
 mkdir OBS_data
 cp -r scratch/traces/syn/* OBS_data/  
 
Do a Full-Waveform Inversion 
Change paths.py to use model_init. We’ll check if wipy_run inversion will recover the true model, so model_true and model_init eventually converge.  
 model_init_path = “/path_to/WIPY/Examples/box/model_init/” 
 obs_data_path = “/path_to/WIPY/Examples/box/OBS_data/” 
 
Run the inversion.  This checks the model against the “observed” data up to the maximum number of iterations, max_inter in parameters.py. 
 wipy_run inversion 
 
View the results in the output_analysis notebook. 
*Note you may have to change the path name. 

