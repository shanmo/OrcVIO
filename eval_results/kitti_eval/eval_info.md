## arg format 

```
    # parse argument
    ap = argparse.ArgumentParser()
    ap.add_argument("-da", "--date", required=False, default=default_date,
                    help="kitti date")
    ap.add_argument("-dr", "--drive", required=False, default=default_drive,
                    help="kitti drive")
    ap.add_argument("-lk", "--load_kitti_det_status", required=False, default=default_lk_status,
                    help="load kitti detection instead of kitti tracklet")
    ap.add_argument("-si", "--start_index", required=False, default=default_start_index,
                    help="start index of kitti")
    ap.add_argument("-ei", "--end_index", required=False, default=default_end_index,
                    help="end index of kitti")
```

```
    if config.load_kitti_det_status == 0:
        self.load_tracklet()
    elif config.load_kitti_det_status == 1:
        self.load_detection()
    else:
        # load nothing for odoemtry
        pass
```

## to compare with CubeSLAM 

- -da 2011_09_26 -dr 0022 -si 0 -ei 800 -lk 0
- -da 2011_09_26 -dr 0023 -si 0 -ei 470 -lk 0
- -da 2011_09_26 -dr 0036 -si 0 -ei 800 -lk 0
- -da 2011_09_26 -dr 0039 -si 0 -ei 395 -lk 0
- -da 2011_09_26 -dr 0061 -si 0 -ei 700 -lk 0
- -da 2011_09_26 -dr 0064 -si 0 -ei 570 -lk 0
- -da 2011_09_26 -dr 0095 -si 0 -ei 265 -lk 1
- -da 2011_09_26 -dr 0096 -si 0 -ei 475 -lk 1
- -da 2011_09_26 -dr 0117 -si 0 -ei 655 -lk 1

## to compare with UCLA paper 

- -da 2011_09_26 -dr 0061 -si 0 -ei 700 -lk 0
- -da 2011_09_26 -dr 0036 -si 0 -ei 800 -lk 0
- -da 2011_09_26 -dr 0022 -si 0 -ei 800 -lk 0
- -da 2011_09_26 -dr 0023 -si 0 -ei 470 -lk 0
- -da 2011_09_26 -dr 0035 -si 0 -ei 125 -lk 1
- -da 2011_09_26 -dr 0039 -si 0 -ei 395 -lk 0
- -da 2011_09_26 -dr 0001 -si 0 -ei 105 -lk 0
- -da 2011_09_26 -dr 0019 -si 0 -ei 475 -lk 0
- -da 2011_09_26 -dr 0064 -si 0 -ei 570 -lk 0
- -da 2011_09_26 -dr 0093 -si 0 -ei 430 -lk 0

## to compare with odometry 

- -da 2011_09_30 -dr 0016 -si 0 -ei 270 -lk 2
- -da 2011_10_03 -dr 0027 -si 0 -ei 4540 -lk 2
- -da 2011_10_03 -dr 0034 -si 0 -ei 4660 -lk 2
- -da 2011_09_30 -dr 0018 -si 0 -ei 2760 -lk 2
- -da 2011_09_30 -dr 0020 -si 0 -ei 1100 -lk 2
- -da 2011_09_30 -dr 0027 -si 0 -ei 1100 -lk 2
- -da 2011_09_30 -dr 0028 -si 1100 -ei 5170 -lk 2 (08: 2011_09_30_drive_0028 001100 005170)
- -da 2011_09_30 -dr 0033 -si 0 -ei 1590 -lk 2
- -da 2011_09_30 -dr 0034 -si 0 -ei 1200 -lk 2
