#!/bin/bash

#make a clean temp folder
mkdir /tmp/Abil
#copy folder but exclude git, docs and dis
rsync -av --progress /home/mv23682/Documents/Abil/ /tmp/Abil/ --exclude .git --exclude dist --exclude docs --exclude studies/wiseman2024/ModelOutput --exclude studies/wiseman2024/data/preprocessing_data/Misc
#copy temp folder to bp
scp -r /tmp/Abil/ mv23682@bc4login.acrc.bris.ac.uk:/user/work/mv23682/
#if copy ok, remove temp folder
rm -rf /tmp/Abil
