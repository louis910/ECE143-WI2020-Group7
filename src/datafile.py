import pandas as pd
import csv
import numpy as np

class dataset:

	mobile = None
	steam = None

	def __init__(self):
		'''
		load raw data from csv files
		'''
		self.mobile = pd.read_csv('../data/appstore_games.csv')
		self.steam = pd.read_csv('../data/steam.csv')


	def process(self):
		'''
		process raw data and save to new csv files
		'''
		self.modify_mobile_age()
		self.modify_mobile_language()
		self.normalize_steam_rating()
		self.steam.fillna(0)
		self.mobile.fillna(0)
		self.add_month_yr_steam()
		self.add_month_yr_mobile()
		self.remove_other_mobile_app()
		self.drop_mobile()
		self.drop_steam()
		self.save_mobile()
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
			average_ratings.append(self.steam['positive_ratings'][num]*5/count)
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

	def add_month_yr_mobile(self):
		'''Input:pd.DataFrame object
		   Output:pd.DataFrame object
		   table=pd.read_csv('fname')
		'''
		assert not self.mobile.empty
		assert isinstance(self.mobile, pd.DataFrame)
	
		yr=[]
		mon=[]
		for item in self.mobile['Original Release Date']:
			mm = item.split('/')[1]
			mon.append(int(mm))
			yy = item.split('/')[2]
			yr.append(int(yy))
		self.mobile['Release Year'] = pd.Series(yr)
		self.mobile['Release Month'] = pd.Series(mon)


   
	def add_month_yr_steam(self):
		'''Input:pd.DataFrame object
		   Output:pd.DataFrame object
		   table=pd.read_csv('fname')
		'''
		assert not self.steam.empty
		assert isinstance(self.steam, pd.DataFrame)
	
		timecol = self.steam['release_date']
		yr = []
		mon = []
		for item in timecol.iteritems():
			ind1 = item[1].find('-')
			ind2 = item[1].rfind('-')
			mm = item[1][ind1+1:ind2]
			mon.append(int(mm))
			yy = item[1][:ind1]
			yr.append(int(yy))

		self.steam['Release Year']=pd.Series(yr)
		self.steam['Release Month']=pd.Series(mon)
	

