import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datafile import dataset

class analysis:
	data = None
	totalprice_mobile = None
	price_steam = None
	releaseTime_steam = None
	averageRating_steam = None
	medianPlaytime_steam = None
	ratingCount_steam = None

	averageRating_mobile = None
	ratingCount_mobile = None
	releaseTime_mobile = None
	price_mobile = None
	
	def __init__(self):
		self.data = dataset()
		self.data.process()
		self.data.add_month_yr_steam()
		self.data.add_month_yr_mobile()
		self.extract_series()
		self.modify_inapp_price()

	def extract_series(self):
		'''
		extract data from dataframes
		'''
		self.price_steam = self.data.steam.get("price")
		self.releaseTime_steam = self.data.steam.get("release_date")
		self.averageRating_steam = self.data.steam.get("average_ratings")
		self.medianPlaytime_steam = self.data.steam.get("median_playtime")
		self.ratingCount_steam = self.data.steam.get("median_playtime")

		self.averageRating_mobile = self.data.mobile.get("Average User Rating")
		self.ratingCount_mobile = self.data.mobile.get("User Rating Count")
		self.releaseTime_mobile = self.data.mobile.get("Original Release Date")
		self.price_mobile = self.data.mobile.get("Price")
		self.price_mobile.astype(float, errors = 'ignore')

		self.yr_month_steam = [str(year)+'-'+str(month) for year,month in zip(self.data.steam["Release Year"], self.data.steam["Release Month"])]
		self.yr_month_mobile = [str(year)+'-'+str(month) for year,month in zip(self.data.mobile["Release Year"], self.data.mobile["Release Month"])]


	def get_column_name(self):
		'''
		print names of columns in dataframes
		'''

		print("The total columns inside steam:")
		print([item for item in self.data.steam.columns])
		print("\nThe total columns inside mobile:")
		print([item for item in self.data.mobile.columns])

	def modify_inapp_price(self):
		"""
		Because the column "in-app purchase has several prices in a line,
		so we calculate the average price and update the original Series.

		"""
		inapp_mobile = self.data.mobile.get("In-app Purchases")
		average_inapp_mobile = []
		for item in inapp_mobile:
		    if pd.notna(item): 
		        item = item.split(',')
		        ave = 0
		        for i in item:
		            ave += float(i)
		        average_inapp_mobile.append(ave/len(item))
		    else:
		        average_inapp_mobile.append(0)

		inapp_mobile = pd.Series(average_inapp_mobile, index=inapp_mobile.index)
		self.data.mobile['inapp_mobile'] = inapp_mobile
		# price + average of in_app purchase price    
		self.totalprice_mobile = self.price_mobile.add(inapp_mobile, fill_value = 0)


	def plot_price_without_inapp(self):
		'''
		Plot of the price of steam vs the price of mobile (w/o in-app purchase)
		blue dot: price of steam
		red dot: price of mobile
		'''

		x = np.array(self.yr_month_steam)
		y = np.array(self.price_steam)
		w = np.array(self.yr_month_mobile)
		z = np.array(self.price_mobile)
		plt.figure(figsize=(18,7))
		plt.axis([0,100,0,50])
		plt.scatter(x, y, alpha = 0.2, label="Steam")
		plt.scatter(w, z, alpha = 0.2, color = 'r', label="Mobile")
		plt.xlabel("Date")
		plt.ylabel("Price")
		plt.title("Prices of Games")
		plt.legend()
		plt.show()

	def plot_price_with_inapp(self):
		'''
		Plot of the price of steam vs the price of mobile (w in-app purchase)
		blue dot: price of steam
		red dot: price of mobile
		'''

		x = np.array(self.yr_month_steam)
		y = np.array(self.price_steam)
		w = np.array(self.yr_month_mobile)
		z = np.array(self.totalprice_mobile)
		plt.figure(figsize=(18,7))
		plt.axis([0,100,0,50])
		plt.scatter(x, y, alpha = 0.2, label = "Steam")
		plt.scatter(w, z, alpha = 0.2, color = 'r', label = "Mobile (with In-app Purchase)")
		plt.xlabel("Date")
		plt.ylabel("Price")
		plt.title("Prices of Games")
		#plt.xticks(np.arange(min(x),max(x),1))
		plt.legend()
		plt.show()

	def number_of_new_games_per_year(self):
		'''
		plot number of launched games in each year
		'''
		count_steam = self.data.steam.set_index(['Release Year','Release Month']).count(level='Release Year')
		launchYear_steam = count_steam.index
		launchNumber_steam = count_steam['name']
		count_mobile = self.data.mobile.set_index(['Release Year', 'Release Month']).count(level='Release Year')
		launchYear_mobile = count_mobile.index
		launchNumber_mobile = count_mobile['Name']
		y = np.array(launchNumber_steam)
		x = np.array(launchYear_steam)
		w = np.array(launchNumber_mobile)
		z = np.array(launchYear_mobile)

		plt.figure(figsize=(15,5))
		plt.plot(x,y)
		plt.plot(z,w, color='r')
		#print(self.data.steam.set_index(['Release Year','Release Month']).count(level='Release Year'))]

	def number_of_new_games_per_month(self):
		'''
		plot the number of launched games in each month
		'''
		count_steam = self.data.steam.set_index(['Release Year','Release Month']).count(level='Release Month')
		launchYear_steam = count_steam.index
		launchNumber_steam = count_steam['name']
		count_mobile = self.data.mobile.set_index(['Release Year', 'Release Month']).count(level='Release Month')
		launchYear_mobile = count_mobile.index
		launchNumber_mobile = count_mobile['Name']
		y = np.array(launchNumber_steam)
		x = np.array(launchYear_steam)
		w = np.array(launchNumber_mobile)
		z = np.array(launchYear_mobile)

		plt.figure(figsize=(15,5))
		plt.plot(x,y)
		plt.plot(z,w, color='r')
		plt.xlabel("Month")
		plt.ylabel("The Number of Games")

	def average_price_in_each_year(self):
		'''
		plot the average price in each year
		'''
		yr_steam = self.date.steam['Release Year'].to_list()
		# remove duplicate items in the list yr_steam
		yr_steam = list(dict.fromkeys(yr_steam))
		averagePrice_steam = []
		for item in yr_steam:
		    averagePrice_steam.append(np.average(self.date.steam[self.date.steam['Release Year']==item]['price']))

		yr_mobile = self.data.mobile['Release Year'].to_list()
		yr_mobile = list(dict.fromkeys(yr_mobile))
		averagePrice_mobile = []
		averageInapp_mobile = []
		for item in yr_mobile:
		    averagePrice_mobile.append(np.average(self.data.mobile[self.data.mobile['Release Year']==item]['Price']))
		    averageInapp_mobile.append(np.average(self.data.mobile[self.data.mobile['Release Year']==item]['inapp_mobile']))
		    

		x = np.array(yr_steam)
		y = np.array(averagePrice_steam)
		w = np.array(yr_mobile)
		z = np.array(averagePrice_mobile)
		n = np.array(averageInapp_mobile)
		plt.figure(figsize=(15,5))
		plt.plot(x,y,label = "Average Price of steam games")
		plt.plot(w,z,color='r', label = "Average Price of mobile game without In-app Purchase")
		plt.plot(w,z+n, color='g', label = "Average Price of mobile game with In-app Purchase")
		plt.xlabel("Release Year")
		plt.ylabel("Price(USD)")
		plt.title("Average Price of Game in each Release Year")
		plt.legend()

	def number_of_game_on_different_platforms(self):
		'''
		plot Number of Game on different Platforms
		'''
		yr_steam = list(set(worker.data.steam['Release Year'].to_list()))
		windows = []
		linux = []
		mac = []
		other = []

		for item in yr_steam[:-1]:
		    w = 0
		    l = 0
		    m = 0
		    o = 0
		    for p in self.data.steam[self.data.steam['Release Year']==item]['platforms']:
		        p.split(';')       
		        for i in p:

		            if p =="windows":
		                w+=1
		            elif p =="mac":
		                m+=1
		            elif p =="linux":
		                l+=1
		            else:
		                o+=1
		    windows.append(w)
		    linux.append(l)
		    mac.append(m)
		    other.append(o)

		plt.figure(figsize=(15,5))
		yr_steam = np.array(yr_steam[:-1])
		linux = np.array(linux)
		windows = np.array(windows)
		mac = np.array(mac)


		#mobile
		yr_mobile = list(set(worker.data.mobile['Release Year'].to_list()))
		mobile = []
		for item in yr_mobile[:-1]:
		    mobile.append(len(self.data.mobile[self.data.mobile['Release Year']==item]))
		    
		#print(mobile)
		yr_mobile = np.array(yr_mobile[:-1])
		mobile = np.array(mobile)


		# plot picture
		plt.plot(yr_steam,linux, color='g', label='Linux')
		plt.plot(yr_steam, mac, color='r', label='Mac')
		plt.plot(yr_steam, windows, label='Windows')
		plt.plot(yr_mobile, mobile, color='lightblue', label = "Mobile")
		plt.legend()
		plt.xlabel("Release Year")
		plt.ylabel('Number of New Games')
		plt.title('Number of Game on different Platforms')
		#self.data.steam["platforms"]

