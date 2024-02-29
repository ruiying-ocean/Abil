

# Running the scripts on BluePebble


To run the script on Bristol BluePebble:

Compile singularity on your machine:

``` sudo singularity build abil.sif Singularity.sif  ```



## Transfer Abil to the cluster.

To upload only relevant files with scp there is a `to-bp.sh` bash script.

To run the script, cd to the planktonSDM directory:

``` cd path_to_planktonSDM ```

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
