#make a clean temp folder
mkdir /tmp/ttt
#copy folder but exclude git and dis
rsync -av --progress /home/phyto/planktonSDM/ /tmp/ttt --exclude .git --exclude dist 
#copy temp folder to bp
scp -r /tmp/ttt ba18321@bp1-login.acrc.bris.ac.uk:/user/work/ba18321/
#if copy ok, remove temp folder
rm -rf /tmp/ttt
