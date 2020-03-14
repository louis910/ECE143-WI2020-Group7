import pandas as pd
import numpy as np
import csv
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from datafile import dataset
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics import silhouette_score


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
	
	mobile_part = None

	def __init__(self):
		self.data = dataset()
		self.data.process()
		self.data.add_month_yr_steam()
		self.data.add_month_yr_mobile()
		self.extract_series()
		self.modify_inapp_price()
		self.mobile_part = self.data.mobile[self.data.mobile['Average User Rating'] > 0]

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
		yr_steam = list(set(self.data.steam['Release Year'].to_list()))
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
		yr_mobile = list(set(self.data.mobile['Release Year'].to_list()))
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

	def plot_rating_vs_count(self):
		'''
		plot figure about average rating score Vs. rating count
		'''

		df_steam = self.data.steam
		plt.figure()
		plt.scatter(df_steam["rating_count"], df_steam["average_ratings"], alpha=0.5)
		plt.xscale("log")
		plt.xlabel("Rating Count")
		plt.ylabel("Average Rating")
		plt.title("Steam: AverageRating - RatingCount")
		plt.savefig("../result/steam_score_count.png")
		plt.show()	

		df_mobile = self.data.mobile
		plt.figure()
		plt.scatter(df_mobile['User Rating Count'], df_mobile['Average User Rating'], alpha=0.5)
		plt.xscale("log")
		plt.xlabel("Rating Count")
		plt.ylabel("Average Rating")
		plt.title("Mobile: AverageRating - RatingCount")
		plt.savefig("../result/mobile_score_count.png")
		plt.show()

	def plot_k_means_SSE(self):

		'''
		plot SSE figure during K-means Clustering with different K
		'''

		clusterdata = np.transpose(np.vstack((self.data.steam['average_ratings'], np.log(self.data.steam['rating_count']))))
		clusterdata_normed = clusterdata / clusterdata.max(axis=0)
		features = clusterdata_normed
		SSE = []  
		Scores = []
		for k in range(2,20):
		    estimator = KMeans(n_clusters=k) 
		    estimator.fit(features)
		    SSE.append(estimator.inertia_)
		X = range(2,20)
		plt.xlabel('k(number of clusters)')
		plt.ylabel('SSE')
		plt.plot(X,SSE,'o-')
		plt.title("SSE curve")
		plt.xticks(np.arange(min(X), max(X)+1, 1.0))
		plt.savefig("../result/cluster_SSE.png")

	def steam_cluster_animation(self):
		'''
		generate animation of clustering on steam dataset
		'''

		clusterdata = np.transpose(np.vstack((self.data.steam['average_ratings'], np.log(self.data.steam['rating_count']))))
		clusterdata_normed = clusterdata / clusterdata.max(axis=0)		
		fig = plt.figure()		
		def update(cnum):
			plt.clf()
			k_means = KMeans(n_clusters=cnum)
			k_means.fit(clusterdata_normed)
			k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0) 
			k_means_labels = pairwise_distances_argmin(clusterdata_normed, k_means_cluster_centers) 
			if (cnum == 10):
				np.save('../result/label10', k_means_labels)
			c = self.data.steam[['average_ratings', 'rating_count']]
			c['cluster'] = k_means_labels
			slices = [c[c['cluster']==i] for i in range(cnum)]
			plots = [plt.scatter(item['rating_count'], item['average_ratings'], alpha=0.3) for item in slices]
			plt.xscale('log')
			plt.title('Number of Clusters: '+str(cnum))
			plt.xlim(0.5, 5000000)
			plt.ylim(-0.5, 5.5)
			return plots, 
		ani = animation.FuncAnimation(fig, update, np.arange(2, 11), interval=1000)
		ffmpegpath = "D:\\ffmpeg\\ffmpeg-20200311-36aaee2-win64-static\\bin\\ffmpeg.exe"
		matplotlib.rcParams["animation.ffmpeg_path"] = ffmpegpath
		writer = animation.FFMpegWriter(fps=1)
		ani.save('../result/cluster.mp4', writer = writer)


	def plot_10_cluster_map(self):
		c = self.data.steam[['average_ratings', 'rating_count']]
		c['cluster'] = np.load('../result/label10.npy')
		slices = [c[c['cluster']==i] for i in range(10)]
		plt.figure()
		for item in slices:
		    plt.scatter(item['rating_count'], item['average_ratings'], alpha=0.3)
		plt.legend(['class '+str(i) for i,item in enumerate(slices)])
		plt.xscale('log')
		plt.xlabel('rating_count')
		plt.ylabel('average_ratings')
		plt.savefig('../result/steam_10.png')

	def steam_generate_success(self):
		'''
		generate success label for steam data
		'''
		self.data.steam['cluster'] = np.load('../result/label10.npy')
		self.data.steam['success'] = (self.data.steam['cluster'] >= 6)
		self.data.steam['success'] = self.data.steam['success'].astype('int')


	
	def plot_steam_success(self):
		c = self.data.steam[['average_ratings', 'rating_count', 'success']]
		c['cluster'] = np.load('../result/label10.npy')
		slices = [c[c['cluster']==i] for i in range(10)]
		plt.figure()
		c0 = c[c['success'] == 0]
		c1 = c[c['success'] == 1]
		plt.scatter(c0['rating_count'], c0['average_ratings'], alpha=0.3)
		plt.scatter(c1['rating_count'], c1['average_ratings'], alpha=0.3)
		plt.xscale('log')
		plt.xlabel('rating_count')
		plt.ylabel('average_ratings')
		plt.legend(['successful', 'not successful'])
		plt.savefig('../result/steam_10_2.png')		
		plt.show()

	def mobile_cluster_animation(self):
		'''
		generate animation of clustering on mobile dataset
		'''

		mobile = self.mobile_part
		clusterdata = np.transpose(np.vstack((0.1+mobile['Average User Rating'], np.log(1+mobile['User Rating Count']))))
		clusterdata_normed = clusterdata / clusterdata.max(axis=0)
		fig = plt.figure()		 
		def update(cnum):
			plt.clf()
			k_means = KMeans(n_clusters=cnum)
			k_means.fit(clusterdata_normed)
			k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0) 
			k_means_labels = pairwise_distances_argmin(clusterdata_normed, k_means_cluster_centers) 
			if (cnum == 10):
				np.save('../result/label10_m', k_means_labels)
			c = mobile[['Average User Rating', 'User Rating Count']]
			c['cluster'] = k_means_labels
			slices = [c[c['cluster']==i] for i in range(cnum)]
			plots = [plt.scatter(item['User Rating Count'], item['Average User Rating'], alpha=0.3) for item in slices]
			plt.xscale('log')
			plt.title('Number of Clusters: '+str(cnum))
			plt.xlim(0.5, 5000000)
			plt.ylim(0, 5.5)
			return plots, 
		ani = animation.FuncAnimation(fig, update, np.arange(2, 11), interval=1000)
		ffmpegpath = "D:\\ffmpeg\\ffmpeg-20200311-36aaee2-win64-static\\bin\\ffmpeg.exe"
		matplotlib.rcParams["animation.ffmpeg_path"] = ffmpegpath
		writer = animation.FFMpegWriter(fps=1)
		ani.save('../result/cluster_m.mp4', writer = writer)

	def mobile_generate_success(self):
		self.mobile_part['cluster'] = np.load('../result/label10_m.npy')
		self.mobile_part['success'] = (self.mobile_part['cluster'] >= 6)
		self.mobile_part['success'] = self.mobile_part['success'].astype('int')

	def plot_mobile_success(self):

		c = self.mobile_part[['User Rating Count','Average User Rating']]
		c.fillna(0)
		c = c[c['Average User Rating'] > 0]
		c['cluster'] = np.load('../result/label10_m.npy')
		slices = [c[c['cluster']==i] for i in range(10)]
		plt.figure()
		c0 = c[c['cluster'] >= 6]
		c1 = c[c['cluster'] < 6]
		plt.scatter(c0['User Rating Count'], c0['Average User Rating'], alpha=0.3)
		plt.scatter(c1['User Rating Count'], c1['Average User Rating'], alpha=0.3)
		plt.xscale('log')
		plt.xlabel('rating_count')
		plt.ylabel('average_ratings')
		plt.legend(['successful', 'not successful'])
		plt.savefig('../result/mobile_10_2.png')
		plt.show()

	def plot_price_Vs_success(self):
		plt.figure()
		plt.hist(self.data.steam[(self.data.steam['success']==0)]['price'], bins=100, alpha=0.5, normed=True)
		plt.hist(self.data.steam[(self.data.steam['success']==1)]['price'], bins=100, alpha=0.3, normed=True)
		plt.ylabel('Percentage')
		plt.xlabel('Price/Dollar')
		plt.title('Steam: Price Distribution')
		plt.xscale('log')
		plt.legend(['successful', 'not successful'])
		plt.savefig('../result/steam_price.png')
		plt.show()
		plt.figure()
		plt.hist(self.mobile_part[(self.mobile_part['success']==0)]['inapp_mobile'], bins=30, alpha=0.5, density=True)
		plt.hist(self.mobile_part[(self.mobile_part['success']==1)]['inapp_mobile'], bins=30, alpha=0.3, density=True)
		plt.ylabel('Percentage')
		plt.xlabel('Price(with In-app)/Dollar')
		plt.title('Mobile: Price Distribution')
		plt.xscale('log')
		plt.legend(['successful', 'not successful'])
		plt.savefig('../result/mobile_price.png')
		plt.show()

	def plot_genres_Vs_success(self):
		steam_g_list_0 = list()
		for item in self.data.steam[self.data.steam['success']==0]['genres']:
		    steam_g_list_0.extend(item.split(';'))
		steam_df0 = pd.DataFrame(pd.value_counts(steam_g_list_0))
		steam_g_list_1 = list()
		for item in self.data.steam[self.data.steam['success']==1]['genres']:
		    steam_g_list_1.extend(item.split(';'))
		steam_df1 = pd.DataFrame(pd.value_counts(steam_g_list_1))
		steam_df = pd.merge(steam_df1,steam_df0,left_index=True,right_index=True,how='inner')
		steam_df.columns = ['successful', 'not successful']
		steam_df['successful']
		steam_df['not successful'] = steam_df['not successful'] / sum(steam_df['not successful'])
		steam_df['successful'] = steam_df['successful'] / sum(steam_df['successful'])
		plt.figure(figsize=(50, 80))
		steam_df.sort_values(by='not successful', ascending=False)[0:15].plot(kind='bar')
		plt.xticks(rotation=75)
		plt.title('Steam: Genres')
		plt.ylabel("Percentage")
		plt.savefig('../result/steam_genres.png', dpi=200, bbox_inches='tight')		
		plt.show()

		mobile_g_list_0 = list()
		for item in self.mobile_part[self.mobile_part['success']==0]['Genres']:
		    mobile_g_list_0.extend(item.split(', '))
		mobile_df0 = pd.DataFrame(pd.value_counts(mobile_g_list_0))
		mobile_g_list_1 = list()
		for item in self.mobile_part[self.mobile_part['success']==1]['Genres']:
		    mobile_g_list_1.extend(item.split(', '))
		mobile_df1 = pd.DataFrame(pd.value_counts(mobile_g_list_1))
		mobile_df = pd.merge(mobile_df1,mobile_df0,left_index=True,right_index=True,how='inner')
		mobile_df.columns = ['successful', 'not successful']
		mobile_df['successful']
		mobile_df['not successful'] = mobile_df['not successful'] / sum(mobile_df['not successful'])
		mobile_df['successful'] = mobile_df['successful'] / sum(mobile_df['successful'])
		plt.figure(figsize=(50, 80))
		mobile_df.sort_values(by='not successful', ascending=False)[1:16].plot(kind='bar')
		plt.xticks(rotation=75)
		plt.title('Mobile: Genres')
		plt.ylabel("Percentage")
		plt.savefig('../result/mobile_genres.png', dpi=200, bbox_inches='tight')
		plt.show()