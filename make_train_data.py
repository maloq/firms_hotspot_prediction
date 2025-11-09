import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import yaml

from sklearn.model_selection import train_test_split
from src.feature_generation.make_features import make_features_and_save
# from src.target_generation.prepare_target_new import print_country_names


if __name__ == "__main__":

    countries_train = ["Dem_Rep_Korea",
                       "Russian_Federation",
                       "Finland",
                       "Norway",
                       "Sweden",
                       "Denmark",
                       "Lithuania",
                       "Latvia",
                       "Estonia",
                       "Poland",
                       "Czech_Republic",
                       "Germany",
                       "Hungary",
                       "Slovakia",
                       "Belarus",
                       "Ukraine",
                       "Moldova",
                       "Romania",
                       "Bulgaria",
                       "Albania",
                       "Montenegro",
                       "Macedonia_Former_Yugoslav_Republic_of",
                       "Kosovo",
                       "Serbia",
                       "Croatia",
                       "Bosnia_and_Herzegovina",
                       "Slovenia",
                       "Greece",
                       "Turkey",
                       "Georgia",
                       "Azerbaijan",
                        "Armenia",
                       "Kazakhstan",
                       "Kyrgyzstan",
                       "Tajikistan",
                       "Mongolia",
                       "China",
                       "Japan",
                       "Republic_of_Korea"]

    for country in countries_train:
        config_path = 'configs/features_config_30d.yaml'
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        config['modis_countries'] = [country]
        config['prediction_countries'] = [country]
        config = dict(config)
        print("skip_climate: ", config['skip_climate'])
        output_file = f"data/saved_features/train_test_features_30d_{country}.parquet"

        make_features_and_save(config, output_file, test_mode=False, use_cached_files=False, use_cached_target=False, cache_dir='data/saved_features/climate_features_cache')

            
        final_df = pd.read_parquet(output_file)
        print(final_df.head())
        print(final_df.shape)
        print(final_df.columns)
        print(final_df.dtypes)
        print(final_df.info())
        print(final_df.describe())
        print(final_df.isnull().sum())
        print(final_df.isnull().sum().sum())
        print(final_df.isnull().sum().sum() / final_df.size)