# ECE 143     Preference in Games: Steam vs. Mobile

Developing a new game costs a lot of money and time for developer, so realizing the customers’ preference is really important before starting to develop. Therefore, we will analyze the trend of games on Steam and mobile devices in the past decade and make comparison between preference of Steam users and mobile users.

## Prerequisites

environment: 
python
ffmpeg

modules:
numpy
pandas
matplotlib
six
seaborn


## Deployment

### ① data

This folder includes two raw datasets from kaggle open datasets.

The steam dataset: https://www.kaggle.com/nikdavis/steam-store-games

The mobile dataset: https://www.kaggle.com/tristan581/17k-apple-app-store-strategy-games

Download and put the '.csv' files under the './data' folder

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

It contains 2 different classes which represent different procedure in our analysis

Here are all functions and their purpose

part1:

1. extract_series: extract data from dataframes

2. get_column_name: print names of columns in dataframes

3. modify_inapp_price: calculate the average of 'in-app purchase' and update the data

4. plot_price_with_inapp_Vs_time: Plot the figure of the price of steam vs the price of mobile games
	- blue dots: price of steam
	- green dots: price of steam with in-app purchase
	- red dots: price of mobile without in-app purchase

5. plot_rating_Vs_count: plot of the average rating score Vs. rating count

6. plot_k_means_SSE: Sum of squared error of K-means Clustering with K differs

7. steam_cluster_animation: generate animation of clustering on steam dataset

8. plot_10_cluster_map: show clustering result when K=10 on steam dataset

9. steam_generate_success: use clustering result generating a new label called 'success' on steam dataset

10. plot_steam_success: plot the new 'success' feature on steam dataset

11. mobile_cluster_animation: generate animation of clustering on mobile dataset

12. mobile_generate_success: use clustering result generating a new label called 'success' on mobile dataset

13. plot_mobile_success: plot the new 'success' feature on mobile dataset

14. plot_price_Vs_success: show the distribution of price according to 'success' label on both datasets

15. plot_genres_Vs_success: show the distribution of popular genres according to 'success' label on both datasets

part2: 

(if not specially mentioned, the functions are applying on both dataset at the same call)

1. steam_rename: rename the columns of steam dataframe in order to be consistent with modile dataframe

2. extract_rate_per_year: extract data with different ratings in every year

3. plot_number_Vs_year: plot the number of new games in every year and use different colors for different ratings

4. plot_rate_percentage: show the percentage of each rating level

5. plot_rating_Vs_price: plot of average rating score Vs. price

6. plot_active_user_Vs_price: plot of active user number Vs. price

7. prepare_data: prepare data for analysis on each genre

8. plot_positive_count: plot of positive rated user count Vs. all rated user count for each genre

9. plot_developed_Vs_active_user: plot of number of games developed Vs. active user count for each genre

10. plot_user_count_Vs_genres: plot of top 10 genres with highest percentage of rating count

11. generate_animation_for_genres: generate animations to demonstrate the change of each genre in different year

### ③ result

This folder includes our figures and animations for the presentation slide. When display figures and animations in main.ipynb, results will be saved here automatically at the same time. 

## Authors (ordered by names)

* **Jiayun Zhou** 
* **Louis Lu** 
* **Yuance Li** 
* **Yue Geng** 
