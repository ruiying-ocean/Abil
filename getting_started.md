Install Miniconda: https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html

Run `conda create -n venv_name` and conda `activate venv_name`, where `venv_name` is the name of your virtual environment.

Install the dependencies by running: `conda install -c conda-forge xarray, pandas, numpy`

Run `conda install pip`. This will install pip to your venv directory.

Find your anaconda directory, and find the actual venv folder. It should be somewhere like `/anaconda/envs/venv_name/`.


Install the planktonSDM packages by doing `/home/phyto/anaconda3/envs/my-geopandas-env-2/bin/pip install path_to_package` where the `path_to_package` is the folder containing the package and should look something like `/home/phyto/planktonSDM/`


Check if the package has been installed correctly by running: `python` and then `import planktonsdm`

 
