# 🗺️ Lumen Geoguesser

<p align="center">
	<img src="readme-pics/geoguesser-logo.png"></img>
</p>


## ⬇️ Setup

Create and populate the [virtual environment](https://docs.python.org/3/library/venv.html#:~:text=A%20virtual%20environment%20is%20a,part%20of%20your%20operating%20system). Simply put, the virtual environment allows you to install Python packages only for this project (which you can easily delete later). This way, we won't clutter your global Python packages.

```bash
[ ! -d "venv" ] && (echo "Creating python3 virtual environment"; python3 -m venv venv)

pip install -r requirements.txt
```

## How to play around and use the files:
### I have the directory `images` that looks like this: Creating enriched dataframe with centroids and regions:

How to 
```
└── images
    ├── 00002003-201f-4863-8677-1860d4a0f828
    │   ├── 0.jpg
    │   ├── 180.jpg
    │   ├── 270.jpg
    │   └── 90.jpg
    ├── 000090c5-90cc-44cb-8f6e-f8ba6e72a73d
    │   ├── 0.jpg
    │   ├── 180.jpg
    │   ├── 270.jpg
    │   └── 90.jpg
    ├── 0000c313-2616-4fe1-9414-b8ae261fb8a2
    │   ├── 0.jpg
    │   ├── 180.jpg
	...
```

## Evaluate:
```sh
curl -X POST lumen.photomath.net/evaluate \
-F 'file=@mapped_to_country_pred-Mike_41-2022-05-06-10-01-15.csv' \
-F "team_code=<INSERT CODE HERE>"
```

Stats:
33.37094934360599 - mapped_to_country_pred-Mike_41-2022-05-06-10-01-15.csv 




## 📁 Directory structure

| Directory                   | Description                    |
| --------------------------- | ------------------------------ |
| [data](./data/)             | dataset                        |
| [models](./models/)         | saved and trained models       |
| [references](./references/) | research papers and guidelines |
| [reports](./reports/)       | model stat's, figures          |
| [src](./src/)               | python source code             |


### Developer notes:

To create `requirements.txt` use the following steps:

```sh
pip install pipreqs
cp requirements.txt requirements.txt.backup
pipreqs --force .
```