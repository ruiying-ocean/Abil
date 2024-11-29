High Performance Computing
==========================


.. code-block:: yaml
    name: abil-env
    channels:
    - defaults
    - conda-forge
    dependencies:
    - netcdf4
    - dask
    - numpy
    - pandas
    - pyyaml
    - xarray
    - xgboost==1.7.6
    - ipykernel
    - scikit-learn==1.3.0
    - scipy
    - mapie
    - scikit-bio



.. code-block:: singularity

    Bootstrap: docker
    From: continuumio/miniconda3

    %files
        ../../dist/abil-0.0.10.tar.gz /root
        ../../examples/conda/environment.yml /root

    %post
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
        . /opt/conda/etc/profile.d/conda.sh
        conda install -n base conda-libmamba-solver
        conda config --set solver libmamba
        conda config --set channel_priority true
        conda config --add channels conda-forge
        conda env update -n base --file /root/environment.yml
        cd
        python -m pip install abil-0.0.10.tar.gz

    %runscript
        . /opt/conda/etc/profile.d/conda.sh
        exec "$@"




# Running the scripts on BluePebble

To run the script on Bristol BluePebble:

Compile singularity on your machine:

First cd to the right folder:

``` cd path_to_Abil/examples/singularity/  ```

If using apptainer (recommended):

``` sudo apptainer build abil.sif Singularity.sif  ```

If using singularity:

``` sudo singularity build abil.sif Singularity.sif  ```

Note: apptainer is often easier to install than singularity and is backwards compatible with legacy Singularity installs:

https://apptainer.org/docs/admin/1.2/installation.html

## Transfer Abil to the cluster.

To upload only relevant files with scp there is a `to-bp.sh` bash script.

To run the script, cd to the Abil directory:

``` cd path_to_Abil ```

Set the bash script executable permission by running chmod command in Linux:

``` sudo chmod +x to-bp.sh ```

Edit the file to your own username.

Then execute it:

``` sudo ./to-bp.sh ``` 

The script will ask for your cluster login information and then transfer the files.



Connect to BluePebble:

``` ssh my_username@bp1-login.acrc.bris.ac.uK ``` 


Change directory to abil:

``` cd /user/work/my_username/abil ``` 


Change directory to folder containing you bash scripts:

``` cd cluster_example ``` 


Submit cluster job:

``` sbatch tune_RF.sh ``` 


Check if the job is running:

``` sacct ``` 


Delete error output files:

``` rm *.out ```
