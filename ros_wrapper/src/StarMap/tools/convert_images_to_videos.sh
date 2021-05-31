for b in data/arl/arl_husky_overpasscity_threecar*.bag; do cd $(pwd)/${b/.bag/} && ffmpeg -framerate 10 -i extract_images_starmap_%04d.jpg -c:v libx264 $(basename ${b/.bag/}.mp4) ; cd -; done
