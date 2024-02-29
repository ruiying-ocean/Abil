#!/bin/bash

#make a clean temp folder
mkdir /tmp/planktonSDM
#copy folder but exclude git, docs and dis
rsync -av --progress /home/phyto/planktonSDM/ /tmp/planktonSDM/ --exclude .git --exclude dist --exclude docs --exclude ModelOutput --exclude devries2024/data --exclude data
#copy temp folder to bp
scp -r /tmp/planktonSDM/ ba18321@bp1-login.acrc.bris.ac.uk:/user/work/ba18321/
#if copy ok, remove temp folder
rm -rf /tmp/planktonSDM
