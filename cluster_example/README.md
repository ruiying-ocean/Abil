



To run the script on Bristol BluePebble:


Compile singularity on your machine:

``` sudo singularity build planktonSDM.sif Singularity.sif  ```

Transfer the whole planktonSDM folder to the cluster (including the singularity container, your yaml files, and data):

``` scp -r  /home/phyto/planktonSDM/  my_username@bp1-login.acrc.bris.ac.uk:/user/work/my_username/```

Connect to BluePebble:
``` ssh my_username@bp1-login.acrc.bris.ac.uK ``` 


Change directory to planktonSDM:

``` cd /user/work/my_username/planktonSDM ``` 


Change directory to folder containing you bash scripts:

``` cd cluster_example ``` 


Submit cluster job:

``` sbatch tune_RF.sh ``` 


Check if the job is running:

``` sacct ``` 
