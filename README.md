  # 🗺️ Lumen Geoguesser

<p align="center">
	<img src="readme-pics/geoguesser-logo.png"></img>
</p>


## 📝 Todo

### NOTE (1): PLEASE EXTRACT YOUR CODE INTO FUNCTIONS 
### NOTE (2): PLEASE ADD DESCRIPTIONS OF THOSE FUNCTIONS.
###       NOTE (2.1) WHAT DOES THE FUNCTION DO?
###       NOTE (2.2) WHAT ARE THE ARGUMENTS?
###       NOTE (2.3) WHAT DOES THE FUNCTION RETURN?
### NOTE (**): IF YOU DON'T DO THIS IN THE MOMENT, YOU ARE JUST LEAVING THE WORK TO SOMEONE IN THE FUTURE.

- [ ] Fix the creation of the grid
  - current situation: each square in the grid is defined as start_lat, start_lng, end_lat, end_lng. This is bad because spacing between those angles is not linear, but we act like they are because we increase the step linearly.
  - [ ] you have to project the lat lngs to a 2D plane before applying the spacing
  - [ ] additionally: replace spacing arg with lenght_of_square_in_meters. Then, via the lenght_of_square_in_meters argument, we will caculate the `spacing` between squares. total number of squares should be approximately lenght_of_square
  - [ ] Before creating the grid, CRS projection should be made!


- [ ] Fix weighted sum classification - Once the dataframe contains projected CRS values, fix the weighted sum of classification predictions so those values are used instead of lat lng values.

- [ ] Generate images that show weights of the model:
  - https://pytorch-lightning-bolts.readthedocs.io/en/latest/vision_callbacks.html#
  - How often should be this called? At the end of every val epoch seems fine
  - [ ] `logging_batch_interval` should be number of batches in 1 epoch (e.g. ~800, this can be calculated)

- [ ] Generate images that shows batch, predicted and true values
  - rules for the task above should apply for this task too
  - I recommend creating a new Callback. https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.Callback.html#pytorch_lightning.callbacks.Callback
  - this callback will hook into (override) `on_validation_batch_end`. After the Callback is passed to Trainer, the Trainer will call Callback's `on_validation_batch_end` at appropriate time.
    - for the example of a Callback check `OnTrainEpochStartLogCallback`
  - Image:
    - [ ] shows all images in the batch. 4 images in first and 4 images in second row
    - [ ] blow the images you can see predicted value and true value for each image
    - [ ] predicted and true values should be expressed as lat lng
    - [ ] title - batch number, additional data whatever


- [ ] Sanity image check - open some images during the training to see if they make sense

- [ ] Outside of Croatia bound classification - prediction gives softmax of values; weighted sum ends up in Bosna, what do we do?
    - Solution 1: find the closest point on the border
    - Solution 2: increase the loss
    - Solution 3: do nothing; model might fit the Croatia borders implicitly

- [ ] Optional: project to CRS (?) then calculate distances, then re-project (?). This is useful if we use regression and non-haversine distance but something linear. By not projecting-reprojecting, the spacing between squares is not correct. Adding/subtracting angles doesn't change affect the distance linearly.

- [ ] Optional: implement polygons on the border; these are additional classes which are explicitly defined. These classes might clash with already exising classes (polygons). How? There might be a polygon which is close to the border and overlaps the explicitly defined polygon. Solution is to remove the intersection so that polygons don't overlap. Polygon on the border (the one that is explicitly defined) should have priority over getting more surface area.

- [x] IMPORTANT: Weighted sum Haversine classification - Whats the current situation? We are making classifications and calling argmax to hard-classify image to a class. Centroid of this class is then used as a prediction. Why are we not taking the weighted sum (softmax probs and centroids) ? 

- [x] WOLF: Angle encoding - we can't use raw angle values in ANY case. We have to transform the angles (both y_true and y_pred) to sensible [0, 1] data. Check this link https://stats.stackexchange.com/questions/218407/encoding-angle-data-for-neural-network and try to find more discussions on similar topic. Sin/cos seems super straight forward but check for other options too.
  - first encode angles via cos/sin
  - then use min-max [0, 1] to scale encoded values
  - apply on y_true and y_pred acordingly
  - decode before caculating the distancec

- [x] Implement the Croatia's CRS projection https://epsg.io/3766
  - note: this projection will transform lat and lng's to a 2D plane which can be used in linear manner
  - [x] in `preprocess_csv_create_classes.py` you have to save the projected values along with angles in the .csv file
  - note:  check `preprocess_sample_coords.py` because there we already used the projection
    - this projection and reprojection is really tricky and in my opinion you should print the values at every step just as a sanity check to make sure everything is working
      - first: SET projection to default crs (4326). To my knowledge, this doesn't change the values yet
      - second: PROJECT by to a new crs (3766)  
      - optional: third: REPROJECT to default (4326) if you need lat lng values again. In the gpd.GeoDataFrame You might even access the original lat,lng without reprojecting but i'm not sure.

## 🧠 Brainstorming

- Create a grid for Croatia. Each square of a grid represents a class. Instead of regression, try classification where these squares will be different classes. Classification should have a probabilistic interpretation. Multiply probabilities (the certainty of each block) to get the final coordinate. Size of the square is a hyperparameter.

- Distance from each coordinate point is not linear. Use the transformation calculates the real world distance between coordinates. The earth is round!

- How do we exploit the fact that the real input is 4 images? We can naively classify all 4 images and then average classifications to get a single coordinate. Is there a better way? Maybe we can concatenate 4 images into a single image (360 view) ? If we can't concatenate images, which model architecture should be taken into account?


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
