#!/bin/bash
cd vid
for d in */; do 
        cd $d
        for file in *.png; do 
                convert $file -resize 600x600 $file; 
        done
        cd ../
done

