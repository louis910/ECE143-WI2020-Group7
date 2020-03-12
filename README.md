# Preference in Games: PC vs. Mobile

Developing a new game costs a lot of money and time for developer, so realizing the customers’ preference is really important before starting to develop. Therefore, we will analyze the trend of games on PC and mobile devices in the past decade and make comparison between preference of PC users and mobile users.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

environment: 
python

modules:
numpy
pandas
matplotlib

## Deployment

### src

This folder includes all our source codes.

##### main.ipynb

All parts of the project are executed here by importing modules.
The notebook shows the ideas, the process and the results of our project.
It would be the first thing to read to know about what we did.

##### datafile.py

This module is used to extract and clean up the dataset.

In order to make two datasets comparable, we removed the contents which are irrelavent or useless, such as no-game apps, IDs or URLs.

Here are all functions and their purpose.

For mobile dataset:

remove_other_mobile_app: remove no-game apps

modify_mobile_age: convert feature 'age' into int type

modify_mobile_language: change 'Languages' column into 1/0 ("0" means non-English, "1" means English)

drop_mobile: delete the useless columns

save_mobile: save mobile data into a new csv file

add_month_yr_mobile: change the original form of dates into years and months

For steam dataset:

normalize_steam_rating: convert binary ratings (positive/ negetive) into quantitative ratings (0-5)

drop_steam: delete the useless columns.

save_steam: save steam data into a new csv file

add_month_yr_steam: change the original form of dates into years and months


## Authors (ordered by names)

* **Jiayun Zhou** 
* **Louis Lu** 
* **Yuance Li** 
* **Yue Geng** 
