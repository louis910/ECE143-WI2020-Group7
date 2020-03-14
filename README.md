# ECE 143     Preference in Games: PC vs. Mobile

Developing a new game costs a lot of money and time for developer, so realizing the customers’ preference is really important before starting to develop. Therefore, we will analyze the trend of games on PC and mobile devices in the past decade and make comparison between preference of PC users and mobile users.

## Prerequisites

environment: 
python

modules:
numpy
pandas
matplotlib

## Deployment

### ① data

This folder includes two raw datasets from kaggle open datasets.

The steam dataset: https://www.kaggle.com/nikdavis/steam-store-games

The mobile dataset: https://www.kaggle.com/tristan581/17k-apple-app-store-strategy-games

### ② src

This folder includes all our source codes.

#### main.ipynb

All parts of the project are executed here by importing modules.
The notebook shows the ideas, the process and the results of our project.
It would be the first thing to read to know about what we did.

#### datafile.py

This module is used to extract and clean up the dataset.

In order to make two datasets comparable, we removed the contents which are irrelavent or useless, such as no-game apps, IDs or URLs.

Here are all functions and their purpose.

For mobile dataset:

1. remove_other_mobile_app: remove no-game apps

2. modify_mobile_age: convert feature 'age' into int type

3. modify_mobile_language: change 'Languages' column into 1/0 ("0" means non-English, "1" means English)

4. drop_mobile: delete the useless columns

5. save_mobile: save mobile data into a new csv file

6. add_month_yr_mobile: change the original form of dates into years and months

For steam dataset:

1. normalize_steam_rating: convert binary ratings (positive/ negetive) into quantitative ratings (0-5)

2. drop_steam: delete the useless columns

3. save_steam: save steam data into a new csv file

4. add_month_yr_steam: change the original form of dates into years and months

#### analysis.py

This module is used to process, analyze and plot the data.

Here are all functions and their purpose

1. extract_series: extract data from dataframes

2. get_column_name: print names of columns in dataframes

3. modify_inapp_price: calculate the average of 'in-app purchase' and update the data

4. plot_price_without_inapp: plot of the price of steam vs the price of mobile (w/o in-app purchase) (blue: steam, red: mobile)

5. plot_price_with_inapp: plot of the price of steam vs the price of mobile (w in-app purchase) (blue: steam, red: mobile)

6. number_of_new_games_per_year: plot number of released games in each year

7. number_of_new_games_per_month: plot the number of released games in each month

8. average_price_in_each_year: plot the average price in each year

9. number_of_game_on_different_platforms: plot Number of Game on different Platforms

### ③ result

This folder includes our figures and tables for the presentation.

## Authors (ordered by names)

* **Jiayun Zhou** 
* **Louis Lu** 
* **Yuance Li** 
* **Yue Geng** 
