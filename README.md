---
# to transform this file to .pdf run the following command: pandoc --standalone --toc  docs/documentation.md --pdf-engine=xelatex --resource-path=docs -o docs/pdf-documentation.pdf

# https://pandoc-discuss.narkive.com/m4QmhNgm/fetch-images-when-creating-pdf
title: Documentation
mainfont: DejaVuSerif.ttf
sansfont: DejaVuSans.ttf
monofont: DejaVuSansMono.ttf 
mathfont: texgyredejavu-math.otf
mainfontoptions:
- Extension=.ttf
- UprightFont=*
- BoldFont=*-Bold
- ItalicFont=*-Italic
- BoldItalicFont=*-BoldItalic
colorlinks: true
linkcolor: red
urlcolor: red
output:
	pdf_document:
		toc: yes
		toc_depth:

geometry: margin=1.2cm
numbersections: true
title: |
	Technical documentation
---


# Lumen Geoguesser

<p align="center">
	<img src="readme-pics/geoguesser-logo.png"></img>
</p>

## Notices:
Although you might be reading this documentation in the form of a PDF file, **we highly recommand that you open the [README.md](README.md) file in a markdown editor** (GitHub, VSCode, PyCharm, IDE...).

As for the API documentation, after setting up the environment, we recommand you run the server with the [`python3 src/app/main.py`](src/app/main.py) command after which you can inspect API endpoints in browser (and execute them too!)

Essentialy, the techincal documentation PDF is rendered from the [README.md](README.md) markdown file and export of the in-browser API documentation. 

Few more notes:
- the documentation assumes you are located at the `.lumen-geoguesser` directory when running Python scripts
- all global variables are defined in [`src/config.py`](src/config.py) and [`src/paths.py`](src/utils_paths.py)
- other directories have their own `README.md` files which are hopefully
- you can run most python files with the `python3 program.py -h` to the sense of which arguments you can/must send and what the script actually does


## 📁 Directory structure

| Directory                   | Description                    |
| --------------------------- | ------------------------------ |
| [data](./data/)             | dataset, csvs, country shapefiles                        |
| [models](./models/)         | model checkpoints, model metadata       |
| [references](./references/) | research papers and competition guidelines |
| [reports](./reports/)       | model stat's, figures          |
| [src](./src/)               | python source code             |


##  Setup

### Virtual environment
Create and populate the [virtual environment](https://docs.python.org/3/library/venv.html#:~:text=A%20virtual%20environment%20is%20a,part%20of%20your%20operating%20system). Simply put, the virtual environment allows you to install Python packages only for this project (which you can easily delete later). This way, we won't clutter your global Python packages.


```bash
[ ! -d "venv" ] && (echo "Creating python3 virtual environment"; python3 -m venv venv)

pip install -r requirements.txt
```

### Dataset setup

The original dataset strucutre has a directory `data` with images and `data.csv` at the top level:
```
dataset_original_subset/
├── data
│   ├── 6bde8efe-a565-4f05-8c60-ae2ffb32ee9b
│   │   ├── 0.jpg
│   │   ├── 180.jpg
│   │   ├── 270.jpg
│   │   └── 90.jpg
│   ├── 6c0ed2ea-b31b-4cfd-9828-4aec22bc0b37
│   │   ├── 0.jpg
│   │   ├── 180.jpg
│   │   ...
│   ...
└── data.csv
```

Before running other scripts you have to properly setup new dataset structure using the [`src/preprocess_setup_datasets.py`](src/preprocess_setup_datasets.py) file. It's important to note that this file accepts multiple dataset directories as an argument and it will make sure to merge the datasets correctly. No changes will be done to your original directories.

```
usage: preprocess_csv_create_rich_static.py [-h] [--csv CSV] [--out dir] [--spacing SPACING] [--out-dir-fig dir] [--fig-format {eps,jpg,jpeg,pdf,pgf,png,ps,raw,rgba,svg,svgz,tif,tiff}] [--no-out]

optional arguments:
  -h, --help            show this help message and exit
  --csv CSV             Dataframe you want to enrich (default: None)
  --out dir             Directory where the enriched dataframe will be saved (default: None)
  --spacing SPACING     Spacing that will be used to create a grid of polygons.
                        Different spacings produce different number of classes
                        0.7 spacing => ~31 classes
                        0.5 spacing => ~55 classes (default: 0.7)
  --out-dir-fig dir     Directory where the figure will be saved (default: /home/matej/projects/lumen-geoguesser/figures)
  --fig-format {eps,jpg,jpeg,pdf,pgf,png,ps,raw,rgba,svg,svgz,tif,tiff}
                        Supported file formats for matplotlib savefig (default: png)
  --no-out              Disable any dataframe or figure saving. Useful when calling inside other scripts (default: False)
```

Example:

```sh
python3 src/preprocess_setup_datasets.py --dataset-dirs data/dataset_original_subset data/dataset_external_subset --out-dir data/dataset_complete_subset
```

To run scripts later, you must transform this structure to the following structure:

```
complete_subset/
├── data.csv
└── data_rich_static__spacing_0.5_classes_55.csv
```

1. The dataset is split into train, val and test directories
2. `data.csv` is csv has concaternated rows of all `data.csv`s
3. _Rich static CSV_ contains region information, which locations (images) are valid etc, centroids...





### I have the directory `images` that looks like this: Creating enriched dataframe with centroids and regions:



## Evaluate:
```sh
curl -X POST lumen.photomath.net/evaluate \
-F 'file=@mapped_to_country_pred-Mike_41-2022-05-06-10-01-15.csv' \
-F "team_code=<INSERT CODE HERE>"
```

Stats:
33.37094934360599 - mapped_to_country_pred-Mike_41-2022-05-06-10-01-15.csv 






### Developer notes:

To create `requirements.txt` use the following steps:

```sh
pip install pipreqs
cp requirements.txt requirements.txt.backup
pipreqs --force .
```


```
run python3 src/train.py --accelerator gpu --devices 1 --num-workers 32 --batch-size 8 --dataset-dir data/raw/ data/external/ --cached-df data/complete/data_huge_spacing_0.21_num_class_211.csv --image-size 224 --lr 0.00002 --unfreeze-at-epoch 1 --scheduler plateau --val_check_interval 0.25 --limit_val_batches 0.4
```

Merging PDFs:
```
pdfunite in-1.pdf in-2.pdf in-n.pdf out.pdf
```