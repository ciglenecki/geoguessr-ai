# 🗺️ Lumen Geoguesser

<p align="center">
	<img src="readme-pics/geoguesser-logo.png"></img>
</p>



## 📝 Tasks

### **(RULE 1): EXTRACT YOUR CODE INTO FUNCTIONS**

### **(RULE 2): ADD DESCRIPTIONS TO YOUR FUNCTIONS.**
### **(RULE 3): IF YOU CAN, DEFINE THE FUNCTION IN utils_functions.py**
## **(!!): IF YOU DON'T DO THIS YOU ARE JUST LEAVING THE WORK TO SOMEONE IN THE FUTURE.**

```py
def function(arg1, arg2) -> List[arg1, arg2]:
  """
  This is a function which adds two elements in the list.

  Args:
    arg1: python object which can be of any type
    arg2: python object which can be of any type
  
  Returns:
    list of length 2 which contains both arguments
  """
```
- [ ] **Fix the creation of the grid in `utils_geo.py` (`def get_grid`) and in places where it's used**
  - current situation: each square in the grid is defined as start_lat, start_lng, end_lat, end_lng. This is bad because spacing between those angles is not linear, but we act like they are because we increase the step linearly.
  - [ ] project the lat lngs to 3766 CRS 2D plane and then use new values (crs_3766_x and crs_3766_y) before applying the spacing. The function _shouldn't_ take care of the projection. Do the projection beforehand.
  - [ ] replace `spacing` arg with `lenght_of_square_in_meters`. Then, via the `lenght_of_square_in_meters` argument, caculate the `spacing` between squares. Total number of squares should be approximately `lenght_of_square`

- [ ] **Fix weighted sum classification** @filipwolf is this done?
  - Once the dataframe contains projected CRS values, fix the weighted sum of classification predictions so those values are used instead of lat lng values.

- [ ] **Set centroids via image distribution, not in the middle**
  - [ ] This logic should live in `data_module_geoguesser.py`
  - new centroids should be caculated only from the Train dataset. Check how itteration over the train dataset**s** is done in `calculate_norm_std.py`
  - this step can't be applied during the creation of the csv dataframe because the dataframe doesn't know which images will be used.
  - current situation: the centroids are set in the middle of the square (or clipped to the border if they are in another country or sea). We should adjust this so that the centroid is **weighted sum of locations of all images IN THE SQUARE**. Notice that we can't take mean of lat/lng, rather we should take mean of the projected values

- [ ] **Generate images that show weights of the model**
  - check: <https://pytorch-lightning-bolts.readthedocs.io/en/latest/vision_callbacks.html>
  - `logging_batch_interval` should be the number of batches in one validation epoch! This can be caculated with `len(val_dataloader)`
  - How often should be this called? At the end of every val epoch

- [ ] **Generate images that shows batch, predicted and true values**
  - rules for the task above should apply for this task too
  - Image:
    - [ ] shows all images in the batch
    - [ ] Make sure it looks pleasing even with batch size of 4 or 32
    - [ ] below the images put text with predicted value and true value for each image
    - [ ] predicted and true values should be expressed as lat lng
    - [ ] title - batch number, additional metadata
  - For this task create a new [pytorch_lightning.callbacks.Callback](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.Callback.html#pytorch_lightning.callbacks.Callback) and call it `ImagePredictionSampler`
  - `ImagePredictionSampler` will override the function `on_validation_batch_end` where you will do the actual logic for saving an image
  - check what arguments you can receive via `on_validation_batch_end`
  - The `ImagePredictionSampler` is passed in the callback list to the `Trainer` in `train.py`
  - The Trainer will execute Callback's `on_validation_batch_end` at the end of the validation batch
  - Custom callback example in our code: `OnTrainEpochStartLogCallback`

- [ ] **Outside of Croatia bound classification** - prediction gives softmax of values; weighted sum ends up in Bosna, what do we do?
  - Solution 1: find the closest point on the border
  - Solution 2: increase the loss
  - Solution 3: do nothing! model might fit the Croatia borders implicitly

- [x] **Implement the Croatia's CRS projection <https://epsg.io/3766>. This projection will transform lat and lng's to a 2D plane which can be used in linear manner**
  - [ ] in `preprocess_csv_create_classes.py` you have to save the projected values along with angles in the .csv file
  - first: SET projection to default crs (4326). To my knowledge, this doesn't change the values yet
  - second: PROJECT by to a new crs (3766)  
  - optional: third: REPROJECT to default (4326) if you need lat lng values again. In the gpd.GeoDataFrame You might even access the original lat,lng without reprojecting but i'm not sure.

- [x] **Weighted sum Haversine classification** - Whats the current situation? We are making classifications and calling argmax to hard-classify image to a class. Centroid of this class is then used as a prediction. Why are we not taking the weighted sum (softmax probs and centroids) ?

- [x] WOLF: Angle encoding - we can't use raw angle values in ANY case. We have to transform the angles (both y_true and y_pred) to sensible [0, 1] data. Check this link <https://stats.stackexchange.com/questions/218407/encoding-angle-data-for-neural-network> and try to find more discussions on similar topic. Sin/cos seems super straight forward but check for other options too.
  - first encode angles via cos/sin
  - then use min-max [0, 1] to scale encoded values
  - apply on y_true and y_pred acordingly
  - decode before caculating the distancec

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

### CSV dataframe

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
