# 🗺️ Lumen Geoguesser

<p align="center">
	<img src="readme-pics/geoguesser-logo.png"></img>
</p>


## ⬇️ Setup

Setup virtual environment:

```bash
[ ! -d "venv" ] && (echo "Creating python3 virtual environment"; python3 -m venv venv)

pip install -r requirements.txt
```

## Evaluate:

curl -X POST lumen.photomath.net/evaluate -F 'file=@mapped_to_country_pred-Mike_41-2022-05-06-10-01-15.csv' -F "team_code=<INSERT CODE HERE>"
Stats:



## 📁 Directory structure

| Directory                   | Description                    |
| --------------------------- | ------------------------------ |
| [data](./data/)             | dataset                        |
| [models](./models/)         | saved and trained models       |
| [references](./references/) | research papers and guidelines |
| [reports](./reports/)       | model stat's, figures          |
| [src](./src/)               | python source code             |
