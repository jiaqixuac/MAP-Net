## HazeWorld

**HazeWorld** is a large-scale synthetic outdoor video dehazing dataset, 
which is built upon [Cityscapes](https://www.cityscapes-dataset.com/),
[DDAD](https://github.com/TRI-ML/DDAD),
[UA-DETRAC](https://detrac-db.rit.albany.edu/),
[VisDrone](https://github.com/VisDrone/VisDrone-Dataset),
[DAVIS](https://davischallenge.org/),
and [REDS](https://seungjunnah.github.io/Datasets/reds.html).
Please refer to these official dataset websites for rights of use.

We use [RCVD](https://robust-cvd.github.io/) to estimate the temporally consistent video depths, which are used to synthesize the hazy videos.
The fog synthesis pipeline is built on [SeeingThroughFog](https://github.com/princeton-computational-imaging/SeeingThroughFog/tree/master/tools/DatasetFoggification).

## Prepare datasets

It is recommended to symlink the dataset root to `$MAP-NET/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

```none
MAP-Net
├── ...
├── data
│   ├── Cityscapes
│   │   ├── leftImg8bit_sequence_trainvaltest
│   │   │   ├── leftImg8bit_sequence
│   │   │   │   ├── train
│   │   │   │   ├── val
│   │   │   │   ├── test
│   ├── HazeWorld
│   │   ├── mapping_hazeworld_cityscapes.txt
│   │   ├── gt
│   │   │   ├── Cityscapes
│   │   │   │   ├── train
│   │   │   │   ├── val
│   │   │   │   ├── test
│   │   │   │   ├── mapping_info_GT_train.txt
│   │   │   │   ├── mapping_info_GT_test.txt
│   │   │   ├── DDAD
│   │   │   │   ├── train
│   │   │   │   ├── val
│   │   │   │   ├── ...
│   │   │   ├── UA-DETRAC
│   │   │   │   ├── train
│   │   │   │   ├── test
│   │   │   │   ├── ...
│   │   │   ├── VisDrone
│   │   │   │   ├── train
│   │   │   │   ├── val
│   │   │   │   ├── test-dev
│   │   │   │   ├── ...
│   │   │   ├── DAVIS
│   │   │   │   ├── train
│   │   │   │   ├── val
│   │   │   │   ├── test-dev
│   │   │   │   ├── test-challenge
│   │   │   │   ├── ...
│   │   │   ├── REDS
│   │   │   │   ├── train
│   │   │   │   ├── val
│   │   │   │   ├── ...
│   │   ├── hazy
│   │   │   ├── ...
│   │   ├── transmission
│   │   │   ├── ...
(symlink)
│   │   ├── train
│   │   │   ├── meta_info_tree_GT_train.json
│   │   │   ├── meta_info_GT_train.txt
│   │   │   ├── meta_info_GT_...
│   │   │   ├── gt (symlink)
│   │   │   │   ├── Cityscapes
│   │   │   │   ├── DDAD
│   │   │   │   ├── UA-DETRAC
│   │   │   │   ├── VisDrone
│   │   │   │   ├── DAVIS
│   │   │   │   ├── REDS
│   │   │   ├── hazy (symlink)
│   │   │   │   ├── ...
│   │   ├── test
│   │   │   ├── meta_info_tree_GT_test.json
│   │   │   ├── meta_info_GT_test.txt
│   │   │   ├── meta_info_GT_...
│   │   │   ├── gt (symlink)
│   │   │   │   ├── ...
│   │   │   ├── hazy (symlink)
│   │   │   │   ├── ...
```

**Step 1.**
Download the data from the links at the bottom.
Since many hazy videos may correspond to one ground-truth video, we adopt the file structure above to save storage.

**Step 2.**
~~Download the [meta files](https://appsrv.cse.cuhk.edu.hk/~jqxu/data/HazeWorld_meta-files.zip) and put them into the corresponding locations (see above).~~
The meta files are provided in the HazeWorld [download link](https://appsrv.cse.cuhk.edu.hk/~jqxu/data/HazeWorld.zip).

**Step 3.**
Symlink the **train** and **test** split using the [script](../tools/data/dehazing/hazeworld/create_symlink_hazeworld.py) and the following command:

```shell
python tools/data/dehazing/hazeworld/create_symlink_hazeworld.py
```

### Cityscapes

The data could be found [here](https://www.cityscapes-dataset.com/downloads/) after registration.
Download [*leftImg8bit_sequence_trainvaltest.zip (324GB)*](https://www.cityscapes-dataset.com/file-handling/?packageID=14).
The used videos can be found in [mapping_hazeworld_cityscapes.txt](https://drive.google.com/file/d/13IZPyeB64lu3szOJsihSPGyUx9cK6yb8/view?usp=share_link).

```shell
python tools/data/dehazing/hazeworld/preprocess_hazeworld_cityscapes.py \
--meta-file data/HazeWorld/mapping_hazeworld_cityscapes.txt \
--input-dir data/Cityscapes --work-dir data/HazeWorld
```

### Others

For others, we provide the [processed data (~100GB)](https://appsrv.cse.cuhk.edu.hk/~jqxu/data/HazeWorld.zip).
You can also refer to their official websites for the original data.

### Notes

* We do some data processing on the original data,
so the numbers and videos may not correspond to the original ones.
Here are some examples,
and more details can be found [here](../tools/data/dehazing/hazeworld/preprocess_hazeworld_cityscapes.py).

   > We sample the frames to keep each video clip of similar length (mostly no more than 100 images per video),
   > using different sampling strategies for each dataset.
   >
   > We use **cv2** to resize (short border to 720 pixels if the original is larger than 720, keeping aspect ratios,
   > and using the default interpolation method) and save (*jpg*, default quality, lossy compression, to save storage) images.

* Also, we manually check the data and remove some improper videos (*e.g.*, indoor or nighttime scenes).
