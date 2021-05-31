for bagfile in data/arl/arl_husky_overpasscity_threecar_640480_?.bag; do
    bagfile_basename=$(pwd)/${bagfile/.bag/}
    roslaunch launch/kitti-sort_ros-starmap.launch bagfile_basename:="${bagfile_basename}";
    ffmpeg -y -framerate 10 -i $bagfile_basename/extract_images_starmap_%04d.jpg -c:v libx264 $bagfile_basename/$(basename "${bagfile_basename}")_starmap.mp4 && \
        rm $bagfile_basename/extract_images_starmap_????.jpg;
done
