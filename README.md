# master-thesis-cybathlon

Master thesis carried out in NCM Lab at ETHZ between March and September 2019.

**Title:** Leveraging Deep Learning for Real-time EEG Classification at Cybathlon's BCI-race

**Author:** Frédéric Debraine

**Supervisors:** Prof. Dr. Nicole Wenderoth, Ernest Mihelj

[Mid-term presentation](https://docs.google.com/presentation/d/1M7-L-o8VcEkF2XUpg1tlzHjlhS4FAOSxXEdc2CnIoFg/edit?usp=sharing)

[Final presentation](https://docs.google.com/presentation/d/1VI9dcGJZGvR_Vj2vGMXPnahVWyrGI8pWVup2zn4acWY/edit?usp=sharing)


## 1. Setup

- ***1. Download Anaconda (Python 3.7 distribution):*** [conda 4.7.12](https://www.anaconda.com/distribution/)
- ***2. Clone/download the current repository***
- ***3. Create Conda environment:***
	- `conda create -n cybathlon`
	- `conda activate cybathlon`
- ***4. Install dependencies:***
	- `pip install -r requirements.txt`
	- `conda install nb_conda_kernels`

## 2. Getting the data

### Datasets
- **BCI Competition 2008 (BCIC IV 2a/b):** [Description 2a](http://www.bbci.de/competition/iv/desc_2a.pdf) - [Description 2b](http://www.bbci.de/competition/iv/desc_2b.pdf) - [Data](http://bbci.de/competition/iv/index.html#download) - [Test labels](http://www.bbci.de/competition/iv/results/index.html#labels)
- **Competition data (Pilots):** Internal

### Folder structure
```
master-thesis-cybathlon
└── code
└── Datasets
    ├── BCI_IV_2a
    │    └── gdf
    │        ├── test
    │        │    ├── labels
    │        │    │    ├── A01E.mat
    │        │    │    ├── ...
    │        │    │    └── A09E.mat
    │        │    ├── A01E.gdf
    │        │    ├── ...
    │        │    └── A09E.gdf
    │        └──  train
    │            ├── A01T.gdf
    │            ├── ...
    │            └── A09T.gdf
    ├── BCI_IV_2b  #(idem)
    └── Pilots
        ├── Pilot_1
        └── Pilot_2
            ├── Session_1
            ├── ...
            └── Session_8
                └── vhdr
                    ├── ***.eeg
                    ├── ***.vhdr
                    └── ***.vmrk
```

## 3. Formatting the data

`conda activate cybathlon`

`jupyter notebook #From ./code/ folder`

Go to `./code/formatting_functions/` folder  and open `examples.ipynb` notebook

You can adapt `save_folder` parameter to modify the folder name in which the .npz files
will be saved (same level as `gdf` or `vhdr` folders).

## 4. Training & validating the models
### Local Jupyter Notebooks

`conda activate cybathlon`

`jupyter notebook #From ./code/ folder`

Go to `./code/offline_pipeline/` folder

Open and run `benchmarking_bcic.ipynb` or `benchmarking_competition_data.ipynb`



### Colab Notebooks

You need to have the `./Datasets/` folder on your Google Drive before running the Colab notebooks. 

The `data_path` parameter should be adapted to your path once your Drive has been mounted.

[Benchmarking - Competition dataset](https://colab.research.google.com/drive/1QLnWBQ0ZXnaVCvoCr--Ro8In2sOHnuE9)

[Benchmarking - BCIC dataset](https://colab.research.google.com/drive/1cRHG0g0a_X-yfjg7U_QXlCQ4idBmjHNm)
