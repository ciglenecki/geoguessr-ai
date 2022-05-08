# Lumen Data Science 2022 Dataset Description
This small document describes the dataset as is provided. For all questions, error reports, or other concerns, please contact the Contact Personell of this challenge.

## The structure of the dataset
The dataset is distributed as a zip file. Inside the zip file are this very document you're reading, and a `data/` directory. There is no `data.csv` file with ground truth as that would defeat the point.

The `data/` directory is quite massive - it's got 4,000 folders with randomly generated names.

### The folders
Every one of the bunch of folders has the same structure: it will always contain 4 images taken in the direction of 0 (North), 90 (East), 180 (South) and 270 (West) degrees of azimuth _from a single location_.

### The objective
The objective is to generate a location for each one of the folders. The output should have the same structure as `data.csv` from the training set. Its columns should be:
1. `uuid` - matches the randomly generated file directory names. All the directory names should be in this column, and all the values in the column should correspond to one existing folder.
2. `latitude` - the first geographic component, measuring the North-South angle between the location on the surface and the equatorial plane.
3. `longitude` - the second geographic component, measuring the East-West angle between the location of the surface and the Greenwich meridian.

## Copyright
- This data was obtained from Google Street View Static API with a special permission by Google. This data is to be used solely for solving the Lumen Data Science 2022 Challenge. Any other usage is in direct conflict with the Terms of Service of eSTUDENT, Photomath and Google. You are instructed, and will be reminded, to get rid of all the data copies after the competition ends. Failure to comply may result in serious consequences.
- The copyright of every photo is written on the photo itself.
