import pandas as pd
import csv
import numpy as np

class dataset:

	steam_data = None
	mobile_data = None
	mobile = None
	steam = None

	def __init__(self):
		'''
		load raw data from csv files
		'''
		raw_m = pd.read_csv('../data/appstore_games.csv')
		raw_s = pd.read_csv('../data/steam.csv')

		self.mobile_raw = raw_m[:]
		self.steam_raw = raw_s[:]
		self.mobile = self.mobile_raw
		self.steam = self.steam_raw

	def process(self):
		'''
		process raw data and save to new csv files
		'''
		self.remove_other_mobile_app()
		self.modify_mobile_age()
		self.modify_mobile_language()
		self.drop_mobile()
		self.save_mobile()
		self.normalize_steam_rating()
		self.drop_steam()
		self.save_steam()

	def remove_other_mobile_app(self):
		'''
		some applications are not "game"
		remove them from our dataset
		'''
		lack_m = set()
		for num, item in enumerate(self.mobile['Primary Genre']):
		    if (item != 'Games'):
		        lack_m.add(num)
		self.mobile = self.mobile.drop(lack_m)
	def modify_mobile_age(self):
		'''
		convert data in Age Rating column to int type
		'''
		for item in list(self.mobile['Age Rating']):
		    item = int(item[0:-1])

	def modify_mobile_language(self):
		'''
		change Languages data into 1/0 form
		"0" means non-English
		"1" means English
		'''
		english = list()
		for item in list(self.mobile['Languages']):
		    if type(item) == float or 'EN' in item:
		        english.append(1)
		    else: 
		        english.append(0)
		self.mobile.insert(2,'english', english)

	def drop_mobile(self):
		'''
		drop columns that we don't care about in mobile data
		'''
		self.mobile = self.mobile.drop(['URL','ID','Subtitle','Icon URL','Description','Size','Primary Genre','Current Version Release Date','Languages'], axis=1)

	def save_mobile(self):
		'''
		save mobile data into a csv file
		'''
		self.mobile.to_csv('../data/mobile_modified.csv')


	def normalize_steam_rating(self):
		'''
		rating for steam games are binary (positive or negetive)
		convert them to 0 or 5 
		(negetive->0, positive->5)
		'''
		average_ratings = list()
		rating_count = list()
		for num in range(len(self.steam)):
		    count = self.steam['positive_ratings'][num]+self.steam['negative_ratings'][num]
		    rating_count.append(count)
		    average_ratings.append(self.steam_raw['positive_ratings'][num]*5/count)
		self.steam.insert(11,'average_ratings', average_ratings)
		self.steam.insert(12,'rating_count', rating_count)


	def drop_steam(self):
		'''
		drop columns that we don't care about in Steam data
		'''
		self.steam = self.steam.drop(['appid','achievements','average_playtime','positive_ratings','negative_ratings'], axis=1)

	def save_steam(self):
		'''
		save steam data into a csv file
		'''
		self.steam.to_csv('../data/steam_modified.csv')
