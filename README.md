# 🗺️ Lumen Geoguesser

<p align="center">
	<img src="readme-pics/geoguesser-logo.png"></img>
</p>

## 🧠 Brainstorming

- Create a grid. Instead of regression, try classificaion where each class is a square on the grid. Classification should have a probabilistic interpretation. Multiply probabilities (the certainty of each block) to get the final coordinate. Size of the square is a hyperparameter.

- Distance from each coordinate point is not linear. Use the transformation caculates the real world distance between cooridantes. The earth is round!


## ⬇️ Setup

Setup virtual environment:
```bash
[ ! -d "venv" ] && (echo "Creating python3 virtual environment"; python3 -m venv venv)

pip install -r requirements.txt
```

## 📁 Directory structure

| Directory                   | Description                    |
| --------------------------- | ------------------------------ |
| [data](./data/)             | dataset                        |
| [models](./models/)         | saved and trained models       |
| [references](./references/) | research papers and guidelines |
| [reports](./reports/)       | model stat's, figures          |
| [src](./src/)               | python source code             |


## 🏆 Team members

<table>
  <tr>
    <td align="center"><a href="URL_HERE"><img src="URL_HERE" width="100px;" alt=""/><br /><sub><b>Borna Katović</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/matejciglenecki"><img src="https://avatars.githubusercontent.com/u/12819849?v=4" width="100px;" alt=""/><br /><sub><b>Matej Ciglenečki</b></sub></a><br /></td>
    <td align="center"><a href="URL_HERE"><img src="URL_HERE" width="100px;" alt=""/><br /><sub><b>Filip Wolf</b></sub></a><br /></td>
</table>