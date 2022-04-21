  # 🗺️ Lumen Geoguesser

<p align="center">
	<img src="readme-pics/geoguesser-logo.png"></img>
</p>

## 🧠 Brainstorming

- Create a grid for Croatia. Each square of a grid represents a class. Instead of regression, try classification where these squares will be different classes. Classification should have a probabilistic interpretation. Multiply probabilities (the certainty of each block) to get the final coordinate. Size of the square is a hyperparameter.

- Distance from each coordinate point is not linear. Use the transformation calculates the real world distance between coordinates. The earth is round!

- How do we exploit the fact that the real input is 4 images? We can naively classify all 4 images and then average classifications to get a single coordinate. Is there a better way? Maybe we can concatenate 4 images into a single image (360 view) ? If we can't concatenate images, which model architecture should be taken into account?

## 📝 Todo

- [x] IMPORTANT: Weighted sum Haversine classification - Whats the current situation? We are making classifications and calling argmax to hard-classify image to a class. Centroid of this class is then used as a prediction. Why are we not taking the weighted sum (softmax probs and centroids) ? 

- Matej: [] Rename reports so that includes basic parameters, also, add random string so that we can reference easy

```
 d =  {
        'A': 'Alpha',  'B': 'Bravo',   'C': 'Charlie',
        'D': 'Delta',  'E': 'Echo',    'F': 'Foxtrot',
        'G': 'Golf',   'H': 'Hotel',   'I': 'India',
        'J': 'Juliett','K': 'Kilo',    'L': 'Lima',
        'M': 'Mike',   'N': 'November','O': 'Oscar',
        'P': 'Papa',   'Q': 'Quebec',  'R': 'Romeo',
        'S': 'Sierra', 'T': 'Tango',   'U': 'Uniform',
        'V': 'Victor', 'W': 'Whiskey', 'X': 'X-ray',
        'Y': 'Yankee', 'Z': 'Zulu'}
Alpha_15, Zulu_99
```

- [] Sanity image check - open some images during the training to see if they make sense

- [] Regression - Create a haversine loss functions by using torch tensor operations (you can't use sklearn). Then, replace the existing loss function. Haversine distance is similar to a residual. It might be useful to square the Haversine distance to get similar formula to MSE.
  - [] Outside of Croatia bound classification - prediction gives softmax of values; weighted sum ends up in Bosna, what do we do?
    - Solution 1: find the closest point on the border
    - Solution 2: increase the loss
    - Solution 3: do nothing; model might fit the Croatia borders implicitly

- [] Haver sine logging - add at the end of the epoch in val, at every step in test and ad end of the epoch in test

- [] monitored value (that we EarlyStop on) should be great-circle distance and not val_loss. This is done by recording haversine as a metric via the self.logger. When hyperparameter is logged it can be used as a metric for EarlyStop.

- [] Optional: project to CRS (?) then calculate distances, then re-project (?). This is useful if we use regression and non-haversine distance but something linear. By not projecting-reprojecting, the spacing between squares is not correct. Adding/subtracting angles doesn't change affect the distance linearly.

- [] Optional: implement polygons on the border; these are additional classes which are explicitly defined. These classes might clash with already exising classes (polygons). How? There might be a polygon which is close to the border and overlaps the explicitly defined polygon. Solution is to remove the intersection so that polygons don't overlap. Polygon on the border (the one that is explicitly defined) should have priority over getting more surface area.

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


## Notes

### CSV dataframe:
uuid, lat, lng, class(polygon_index)

### Lightning
[Multiple Datasets
](https://pytorch-lightning.readthedocs.io/en/stable/guides/data.html#multiple-datasets)

There are a few ways to pass multiple Datasets to Lightning:
- Create a DataLoader that iterates over multiple Datasets under the hood.
- In the training loop you can pass multiple DataLoaders as a dict or list/tuple and Lightning will automatically combine the batches from different DataLoaders.
- In the validation and test loop you have the option to return multiple DataLoaders, which Lightning will call sequentially.

## 🏆 Team members

<table>
  <tr>
    <td align="center"><a href="https://github.com/bkatovic"><img src="https://avatars.githubusercontent.com/u/56589395?v=4" width="100px;" alt=""/><br /><sub><b>Borna Katović</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/matejciglenecki"><img src="https://avatars.githubusercontent.com/u/12819849?v=4" width="100px;" alt=""/><br /><sub><b>Matej Ciglenečki</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/filipwolf"><img src="https://avatars.githubusercontent.com/u/50752058?v=4" width="100px;" alt=""/><br /><sub><b>Filip Wolf</b></sub></a><br /></td>
</table>
