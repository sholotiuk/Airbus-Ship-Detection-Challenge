import os
from src.config import *
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import make_image_gen, create_aug_gen, ImageDataGenerator

# Defining relation paths
# BASE_DIR is the path to dataset folder
BASE_DIR = '..input/'
TRAIN_DIR = BASE_DIR + 'train_v2/'
TEST_DIR = BASE_DIR + 'test_v2/'

train = os.listdir(TRAIN_DIR)
test = os.listdir(TEST_DIR)

def main():
    # Data preprocessing steps
    masks = pd.read_csv(os.path.join(BASE_DIR, 'train_ship_segmentations_v2.csv'))
    not_empty = pd.notna(masks.EncodedPixels)
    print(not_empty.sum(), 'masks in', masks[not_empty].ImageId.nunique(), 'images')
    print((~not_empty).sum(), 'empty images in', masks.ImageId.nunique(), 'total images')
    masks.head()

    masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
    unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x > 0 else 0.0)
    masks.drop(['ships'], axis=1, inplace=True)

    # Empty images under-sampling process
    SAMPLES_PER_GROUP = 4000
    balanced_train_df = unique_img_ids.groupby('ships').apply(
        lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x)

    # Creating the validation set
    train_ids, valid_ids = train_test_split(balanced_train_df,
                                            test_size=0.2,
                                            stratify=balanced_train_df['ships'])
    train_df = pd.merge(masks, train_ids)
    valid_df = pd.merge(masks, valid_ids)
    print(train_df.shape[0], 'training masks')
    print(valid_df.shape[0], 'validation masks')

    # Defining the image generators
    train_gen = make_image_gen(train_df)
    train_x, train_y = next(train_gen)
    valid_x, valid_y = next(make_image_gen(valid_df, VALID_IMG_COUNT))

    # Data Augmentation Process
    dg_args = dict(featurewise_center=False,
                   samplewise_center=False,
                   rotation_range=45,
                   width_shift_range=0.1,
                   height_shift_range=0.1,
                   shear_range=0.01,
                   zoom_range=[0.9, 1.25],
                   horizontal_flip=True,
                   vertical_flip=True,
                   fill_mode='reflect',
                   data_format='channels_last')

    if AUGMENT_BRIGHTNESS:
        dg_args[' brightness_range'] = [0.5, 1.5]
    image_gen = ImageDataGenerator(**dg_args)

    if AUGMENT_BRIGHTNESS:
        dg_args.pop('brightness_range')
    label_gen = ImageDataGenerator(**dg_args)

    cur_gen = create_aug_gen(train_gen)
    t_x, t_y = next(cur_gen)
    return train_df, valid_x, valid_y

if __name__ == "__main__":
    train_df, valid_x, valid_y = main()