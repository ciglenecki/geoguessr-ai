import torch


def _get_class_to_centroid_list_cart(df, num_classes: int):
    """
    Args:
        num_classes: number of classes that were recounted ("y" column)
    Itterate over the information of each valid polygon/class and return it's centroids
    """

    df_class_info = df.loc[
        :,
        [
            "polygon_index",
            "y",
            "centroid_x",
            "centroid_y",
            "centroid_z",
            "is_true_centroid",
        ],
    ].drop_duplicates()
    _class_to_crs_centroid_map = []
    for class_idx in range(num_classes):
        row = df_class_info.loc[df_class_info["y"] == class_idx].head(1)  # ensure that only one row is taken
        polygon_x, polygon_y, polygon_z = (
            row["centroid_x"].values[0],
            row["centroid_y"].values,
            row["centroid_z"].values[0],
        )  # values -> ndarray with 1 dim
        point = [polygon_x, polygon_y, polygon_z]
        _class_to_crs_centroid_map.append(point)
    return _class_to_crs_centroid_map


def coords_transform_cart(self, x, y, z):

    min_max_x = (x - self.x_min) / self.x_max
    min_max_y = (y - self.y_min) / self.y_max
    min_max_z = (z - self.z_min) / self.z_max

    return torch.tensor([min_max_x, min_max_y, min_max_z]).float()


def cart_to_lat_long(self, y, images):

    y[:, 0] = y[:, 0] * self.datamodule.x_max + self.datamodule.x_min
    y[:, 1] = y[:, 1] * self.datamodule.y_max + self.datamodule.y_min
    y[:, 2] = y[:, 2] * self.datamodule.z_max + self.datamodule.z_min

    tmp_tensor = torch.zeros(y.size(0), y.size(1))
    tmp_tensor[:, :2] = y[:, [0, 1]] ** 2
    latitude_y = torch.atan2(y[:, 2], torch.sum(tmp_tensor[:, :2], dim=-1).sqrt())
    longitude_y = torch.atan2(y[:, 1], y[:, 0])

    images[:, 0] = images[:, 0] * self.datamodule.x_max + self.datamodule.x_min
    images[:, 1] = images[:, 1] * self.datamodule.y_max + self.datamodule.y_min
    images[:, 2] = images[:, 2] * self.datamodule.z_max + self.datamodule.z_min

    tmp_tensor = torch.zeros(images.size(0), images.size(1))

    tmp_tensor[:, :2] = images[:, [0, 1]] ** 2
    latitude_images = torch.atan2(images[:, 2], torch.sum(tmp_tensor[:, :2], dim=-1).sqrt())
    longitude_images = torch.atan2(images[:, 1], images[:, 0])

    return torch.stack((latitude_y, longitude_y)), torch.stack((latitude_images, longitude_images))
