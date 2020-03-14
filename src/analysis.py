import pandas as pd
import numpy as np
import csv
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datafile import dataset
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics import silhouette_score
import itertools as it 
import functools as func
import random as rd
from pandas.plotting import table
from IPython.display import HTML
import seaborn as sns


class part1:
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
		self.extract_series()
		self.mobile_modify_inapp_price()
		self.mobile_part = self.data.mobile[self.data.mobile["Average User Rating"] > 0]

	def extract_series(self):
		"""
		extract data from dataframes
		"""
		self.price_steam = self.data.steam.get("price")
		self.releaseTime_steam = self.data.steam.get("release_date")
		self.averageRating_steam = self.data.steam.get("average_ratings")
		self.medianPlaytime_steam = self.data.steam.get("median_playtime")
		self.ratingCount_steam = self.data.steam.get("median_playtime")

		self.averageRating_mobile = self.data.mobile.get("Average User Rating")
		self.ratingCount_mobile = self.data.mobile.get("User Rating Count")
		self.releaseTime_mobile = self.data.mobile.get("Original Release Date")
		self.price_mobile = self.data.mobile.get("Price")
		self.price_mobile.astype(float, errors = "ignore")

		self.yr_month_steam = [str(year)+"-"+str(month) for year,month in zip(self.data.steam["Release Year"], self.data.steam["Release Month"])]
		self.yr_month_mobile = [str(year)+"-"+str(month) for year,month in zip(self.data.mobile["Release Year"], self.data.mobile["Release Month"])]


	def get_column_name(self):
		"""
		print names of columns in dataframes
		"""

		print("The total columns inside steam:")
		print([item for item in self.data.steam.columns])
		print("\nThe total columns inside mobile:")
		print([item for item in self.data.mobile.columns])

	def mobile_modify_inapp_price(self):
		"""
		Because the column "in-app purchase has several prices in a line,
		so we calculate the average price and update the original Series.

		"""
		inapp_mobile = self.data.mobile.get("In-app Purchases")
		average_inapp_mobile = []
		for item in inapp_mobile:
			if pd.notna(item): 
				item = item.split(",")
				ave = 0
				for i in item:
					ave += float(i)
				average_inapp_mobile.append(ave/len(item))
			else:
				average_inapp_mobile.append(0)

		inapp_mobile = pd.Series(average_inapp_mobile, index=inapp_mobile.index)
		self.data.mobile["inapp_mobile"] = inapp_mobile
		# price + average of in_app purchase price	
		self.totalprice_mobile = self.price_mobile.add(inapp_mobile, fill_value = 0)


	def plot_price_with_inapp_Vs_time(self):
		"""
		Plot the figure of the price of steam vs the price of mobile (w in-app purchase)
		:blue dots: price of steam
		:green dots: price of steam with in-app purchase
		:red dots: price of mobile without in-app purchase
		"""

		date_steam = np.array(self.data.steam["Release Year"] + self.data.steam["Release Month"]/12)
		price_steam = np.array(self.price_steam)
		date_mobile = np.array(self.data.mobile["Release Year"] + self.data.mobile["Release Month"]/12)
		price_mobile = np.array(self.price_mobile)
		total_price_mobile = np.array(self.totalprice_mobile)

		# generate new x ticks
		new_xtick = []
		start_year = 1997
		for i in range(180):
			if i%8==0:
				new_xtick.append(start_year)
				start_year +=1
			else:
				new_xtick.append("")

		# plot figure		
		plt.figure(figsize=(18, 7))
		plt.ylim(0, 50)
		plt.scatter(date_steam, price_steam, alpha=0.2, label="Steam")
		plt.scatter(date_mobile, total_price_mobile, alpha=0.2, color="g", label="Mobile (with In-app purchase)")
		plt.scatter(date_mobile, price_mobile, alpha = 0.2, color = "r", label = "Mobile (without In-app purchase)")
		#plt.xticks(np.arange(180), new_xtick)
		plt.xlabel("Year")
		plt.ylabel("Price")
		plt.title("Prices of Games")
		plt.legend()
		plt.savefig("../result/price_Vs_year.png")
		plt.show()

	def plot_rating_Vs_count(self):
		"""
		plot figure about average rating score Vs. rating count
		"""

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
		plt.scatter(df_mobile["User Rating Count"], df_mobile["Average User Rating"], alpha=0.5)
		plt.xscale("log")
		plt.xlabel("Rating Count")
		plt.ylabel("Average Rating")
		plt.title("Mobile: AverageRating - RatingCount")
		plt.savefig("../result/mobile_score_count.png")
		plt.show()

	def plot_k_means_SSE(self):

		"""
		plot SSE figure during K-means Clustering with different K
		"""

		clusterdata = np.transpose(np.vstack((self.data.steam["average_ratings"], np.log(self.data.steam["rating_count"]))))
		clusterdata_normed = clusterdata / clusterdata.max(axis=0)
		features = clusterdata_normed
		SSE = []  
		Scores = []
		for k in range(2,20):
			estimator = KMeans(n_clusters=k) 
			estimator.fit(features)
			SSE.append(estimator.inertia_)
		X = range(2,20)
		plt.figure()
		plt.plot(X,SSE,"o-")
		plt.xlabel("k(number of clusters)")
		plt.ylabel("SSE")
		plt.title("SSE curve")
		plt.xticks(np.arange(min(X), max(X)+1, 1.0))
		plt.savefig("../result/cluster_SSE.png")

	def steam_cluster_animation(self):
		"""
		generate animation of clustering on steam dataset
		"""

		clusterdata = np.transpose(np.vstack((self.data.steam["average_ratings"], np.log(self.data.steam["rating_count"]))))
		clusterdata_normed = clusterdata / clusterdata.max(axis=0)		
		fig = plt.figure()		
		def update(cnum):
			plt.clf()
			k_means = KMeans(n_clusters=cnum)
			k_means.fit(clusterdata_normed)
			k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0) 
			k_means_labels = pairwise_distances_argmin(clusterdata_normed, k_means_cluster_centers) 
			if (cnum == 10):
				np.save("../result/label10", k_means_labels)
			c = self.data.steam[["average_ratings", "rating_count"]]
			c["cluster"] = k_means_labels
			slices = [c[c["cluster"]==i] for i in range(cnum)]
			plots = [plt.scatter(item["rating_count"], item["average_ratings"], alpha=0.3) for item in slices]
			plt.xscale("log")
			plt.title("Number of Clusters: "+str(cnum))
			plt.xlim(0.5, 5000000)
			plt.ylim(-0.5, 5.5)
			return plots, 
		ani = animation.FuncAnimation(fig, update, np.arange(2, 11), interval=1000)
		# ffmpegpath = "D:\\ffmpeg\\ffmpeg-20200311-36aaee2-win64-static\\bin\\ffmpeg.exe"
		# matplotlib.rcParams["animation.ffmpeg_path"] = ffmpegpath
		writer = animation.FFMpegWriter(fps=1)
		ani.save("../result/cluster.mp4", writer = writer)


	def plot_10_cluster_map(self):
		c = self.data.steam[["average_ratings", "rating_count"]]
		c["cluster"] = np.load("../result/label10.npy")
		slices = [c[c["cluster"]==i] for i in range(10)]
		plt.figure()
		for item in slices:
			plt.scatter(item["rating_count"], item["average_ratings"], alpha=0.3)
		plt.legend(["class "+str(i) for i,item in enumerate(slices)])
		plt.xscale("log")
		plt.xlabel("rating_count")
		plt.ylabel("average_ratings")
		plt.savefig("../result/steam_10.png")

	def steam_generate_success(self):
		"""
		generate success label for steam data
		"""
		self.data.steam["cluster"] = np.load("../result/label10.npy")
		self.data.steam["success"] = (self.data.steam["cluster"] >= 6)
		self.data.steam["success"] = self.data.steam["success"].astype("int")


	
	def plot_steam_success(self):
		c = self.data.steam[["average_ratings", "rating_count", "success"]]
		c["cluster"] = np.load("../result/label10.npy")
		slices = [c[c["cluster"]==i] for i in range(10)]
		plt.figure()
		c0 = c[c["success"] == 0]
		c1 = c[c["success"] == 1]
		plt.scatter(c0["rating_count"], c0["average_ratings"], alpha=0.3)
		plt.scatter(c1["rating_count"], c1["average_ratings"], alpha=0.3)
		plt.xscale("log")
		plt.xlabel("rating_count")
		plt.ylabel("average_ratings")
		plt.legend(["successful", "not successful"])
		plt.savefig("../result/steam_10_2.png")		
		plt.show()

	def mobile_cluster_animation(self):
		"""
		generate animation of clustering on mobile dataset
		"""

		mobile = self.mobile_part
		clusterdata = np.transpose(np.vstack((0.1+mobile["Average User Rating"], np.log(1+mobile["User Rating Count"]))))
		clusterdata_normed = clusterdata / clusterdata.max(axis=0)
		fig = plt.figure()		 
		def update(cnum):
			plt.clf()
			k_means = KMeans(n_clusters=cnum)
			k_means.fit(clusterdata_normed)
			k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0) 
			k_means_labels = pairwise_distances_argmin(clusterdata_normed, k_means_cluster_centers) 
			if (cnum == 10):
				np.save("../result/label10_m", k_means_labels)
			c = mobile[["Average User Rating", "User Rating Count"]]
			c["cluster"] = k_means_labels
			slices = [c[c["cluster"]==i] for i in range(cnum)]
			plots = [plt.scatter(item["User Rating Count"], item["Average User Rating"], alpha=0.3) for item in slices]
			plt.xscale("log")
			plt.title("Number of Clusters: "+str(cnum))
			plt.xlim(0.5, 5000000)
			plt.ylim(0, 5.5)
			return plots, 
		ani = animation.FuncAnimation(fig, update, np.arange(2, 11), interval=1000)
		# ffmpegpath = "D:\\ffmpeg\\ffmpeg-20200311-36aaee2-win64-static\\bin\\ffmpeg.exe"
		# matplotlib.rcParams["animation.ffmpeg_path"] = ffmpegpath
		writer = animation.FFMpegWriter(fps=1)
		ani.save("../result/cluster_m.mp4", writer = writer)

	def mobile_generate_success(self):
		self.mobile_part["cluster"] = np.load("../result/label10_m.npy")
		self.mobile_part["success"] = (self.mobile_part["cluster"] >= 6)
		self.mobile_part["success"] = self.mobile_part["success"].astype("int")

	def plot_mobile_success(self):

		c = self.mobile_part[["User Rating Count","Average User Rating"]]
		c.fillna(0)
		c = c[c["Average User Rating"] > 0]
		c["cluster"] = np.load("../result/label10_m.npy")
		slices = [c[c["cluster"]==i] for i in range(10)]
		plt.figure()
		c0 = c[c["cluster"] >= 6]
		c1 = c[c["cluster"] < 6]
		plt.scatter(c0["User Rating Count"], c0["Average User Rating"], alpha=0.3)
		plt.scatter(c1["User Rating Count"], c1["Average User Rating"], alpha=0.3)
		plt.xscale("log")
		plt.xlabel("rating_count")
		plt.ylabel("average_ratings")
		plt.legend(["successful", "not successful"])
		plt.savefig("../result/mobile_10_2.png")
		plt.show()

	def plot_price_Vs_success(self):
		plt.figure()
		plt.hist(self.data.steam[(self.data.steam["success"]==0)]["price"], bins=100, alpha=0.5, normed=True)
		plt.hist(self.data.steam[(self.data.steam["success"]==1)]["price"], bins=100, alpha=0.3, normed=True)
		plt.ylabel("Percentage")
		plt.xlabel("Price/Dollar")
		plt.title("Steam: Price Distribution")
		plt.xscale("log")
		plt.legend(["successful", "not successful"])
		plt.savefig("../result/steam_price.png")
		plt.show()
		plt.figure()
		plt.hist(self.mobile_part[(self.mobile_part["success"]==0)]["inapp_mobile"], bins=30, alpha=0.5, density=True)
		plt.hist(self.mobile_part[(self.mobile_part["success"]==1)]["inapp_mobile"], bins=30, alpha=0.3, density=True)
		plt.ylabel("Percentage")
		plt.xlabel("Price(with In-app)/Dollar")
		plt.title("Mobile: Price Distribution")
		plt.xscale("log")
		plt.legend(["successful", "not successful"])
		plt.savefig("../result/mobile_price.png")
		plt.show()

	def plot_genres_Vs_success(self):
		steam_g_list_0 = list()
		for item in self.data.steam[self.data.steam["success"]==0]["genres"]:
			steam_g_list_0.extend(item.split(";"))
		steam_df0 = pd.DataFrame(pd.value_counts(steam_g_list_0))
		steam_g_list_1 = list()
		for item in self.data.steam[self.data.steam["success"]==1]["genres"]:
			steam_g_list_1.extend(item.split(";"))
		steam_df1 = pd.DataFrame(pd.value_counts(steam_g_list_1))
		steam_df = pd.merge(steam_df1,steam_df0,left_index=True,right_index=True,how="inner")
		steam_df.columns = ["successful", "not successful"]
		steam_df["successful"]
		steam_df["not successful"] = steam_df["not successful"] / sum(steam_df["not successful"])
		steam_df["successful"] = steam_df["successful"] / sum(steam_df["successful"])
		plt.figure(figsize=(50, 80))
		steam_df.sort_values(by="not successful", ascending=False)[0:15].plot(kind="bar")
		plt.xticks(rotation=75)
		plt.title("Steam: Genres")
		plt.ylabel("Percentage")
		plt.savefig("../result/steam_genres.png", dpi=200, bbox_inches="tight")		
		plt.show()

		mobile_g_list_0 = list()
		for item in self.mobile_part[self.mobile_part["success"]==0]["Genres"]:
			mobile_g_list_0.extend(item.split(", "))
		mobile_df0 = pd.DataFrame(pd.value_counts(mobile_g_list_0))
		mobile_g_list_1 = list()
		for item in self.mobile_part[self.mobile_part["success"]==1]["Genres"]:
			mobile_g_list_1.extend(item.split(", "))
		mobile_df1 = pd.DataFrame(pd.value_counts(mobile_g_list_1))
		mobile_df = pd.merge(mobile_df1,mobile_df0,left_index=True,right_index=True,how="inner")
		mobile_df.columns = ["successful", "not successful"]
		mobile_df["successful"]
		mobile_df["not successful"] = mobile_df["not successful"] / sum(mobile_df["not successful"])
		mobile_df["successful"] = mobile_df["successful"] / sum(mobile_df["successful"])
		plt.figure(figsize=(50, 80))
		mobile_df.sort_values(by="not successful", ascending=False)[1:16].plot(kind="bar")
		plt.xticks(rotation=75)
		plt.title("Mobile: Genres")
		plt.ylabel("Percentage")
		plt.savefig("../result/mobile_genres.png", dpi=200, bbox_inches="tight")
		plt.show()


class part2:

	Data_steam = None
	Data_mobile = None

	steam_rate_2 = None
	steam_rate_3 = None
	steam_rate_4 = None
	steam_rate_5 = None

	mobile_rate_2 = None
	mobile_rate_3 = None
	mobile_rate_4 = None
	mobile_rate_5 = None

	def __init__(self):

		self.Data_steam = pd.read_csv("../data/steam_modified.csv")
		self.Data_mobile = pd.read_csv("../data/mobile_modified.csv")
		self.steam_rename()
		self.extract_rate_per_year()

	def steam_rename(self):
		self.Data_steam.rename(columns={"genres":"Genres","rating_count":"User Rating Count","release_date":"Original Release Date","average_ratings":"Average User Rating"},inplace=True)

	def extract_rate_per_year(self):
		self.Data_mobile["Counter yr/rating"]=1
		Data_mobile_v=self.Data_mobile.groupby(["Release Year","Average User Rating"])["Counter yr/rating"].sum()
		#Analyze price-related stat
		self.mobile_rate_2=np.zeros(12)
		self.mobile_rate_3=np.zeros(12)
		self.mobile_rate_4=np.zeros(12)
		self.mobile_rate_5=np.zeros(12)
		yrs=list(range(2008,2020))
		for i in range(len(Data_mobile_v.index)):
			for yr in range(len(yrs)):
				if Data_mobile_v.index[i][0]==yrs[yr] and Data_mobile_v.index[i][1]<=2:
					self.mobile_rate_2[yr]=self.mobile_rate_2[yr]+Data_mobile_v[i]
				elif Data_mobile_v.index[i][0]==yrs[yr] and Data_mobile_v.index[i][1]<=3:
					self.mobile_rate_3[yr]=self.mobile_rate_3[yr]+Data_mobile_v[i]
				elif Data_mobile_v.index[i][0]==yrs[yr] and Data_mobile_v.index[i][1]<=4:
					self.mobile_rate_4[yr]=self.mobile_rate_4[yr]+Data_mobile_v[i]
				elif Data_mobile_v.index[i][0]==yrs[yr] and Data_mobile_v.index[i][1]<=5:
					self.mobile_rate_5[yr]=self.mobile_rate_5[yr]+Data_mobile_v[i]

		self.Data_steam["Counter yr/rating"]=1
		self.Data_steam["Average User Rating"]=self.Data_steam["Average User Rating"].round(3)
		Data_steam_v=self.Data_steam.groupby(["Release Year","Average User Rating"])["Counter yr/rating"].sum().reset_index()
		#Analyze price-related stat
		self.steam_rate_2=np.zeros(12)
		self.steam_rate_3=np.zeros(12)
		self.steam_rate_4=np.zeros(12)
		self.steam_rate_5=np.zeros(12)
		yrs=list(range(2008,2020))
		for i in range(len(Data_steam_v)):
			for yr in range(len(yrs)):
				if Data_steam_v["Release Year"][i]==yrs[yr] and Data_steam_v["Average User Rating"][i]<=2:
					self.steam_rate_2[yr]=self.steam_rate_2[yr]+Data_steam_v["Counter yr/rating"][i]
				elif Data_steam_v["Release Year"][i]==yrs[yr] and Data_steam_v["Average User Rating"][i]<=3:
					self.steam_rate_3[yr]=self.steam_rate_3[yr]+Data_steam_v["Counter yr/rating"][i]
				elif Data_steam_v["Release Year"][i]==yrs[yr] and Data_steam_v["Average User Rating"][i]<=4:
					self.steam_rate_4[yr]=self.steam_rate_4[yr]+Data_steam_v["Counter yr/rating"][i]
				elif Data_steam_v["Release Year"][i]==yrs[yr] and Data_steam_v["Average User Rating"][i]<=5:
					self.steam_rate_5[yr]=self.steam_rate_5[yr]+Data_steam_v["Counter yr/rating"][i]



	def plot_number_Vs_year(self):
		sns.set()
		yrs=list(range(2008,2020))
		bardata_s=pd.DataFrame(index=yrs,data={"2 or below":self.steam_rate_2,"2 to 3":self.steam_rate_3,"3 to 4":self.steam_rate_4,"4 to 5":self.steam_rate_5})	   
		plt.figure(figsize=(8,6))
		bardata_s.plot(kind="barh",stacked=True,colormap="Accent")
		plt.title("Steam",size=20)
		plt.savefig("../result/Number of Games with Rating by Year_steam.png")
		plt.show()

		bardata_m=pd.DataFrame(index=yrs,data={"2 or below":self.mobile_rate_2,"2 to 3":self.mobile_rate_3,"3 to 4":self.mobile_rate_4,"4 to 5":self.mobile_rate_5})	   
		plt.figure()
		bardata_m.plot(kind="barh",stacked=True,colormap="Accent")
		plt.title("Mobile",size=20)
		plt.savefig("../result/Number of Games with Rating by Year_mobile.png")
		plt.show()

	def plot_rate_percentage(self):
		sns.set()
		plt.figure(figsize=(8,6))
		pielabels = "2 or below", "2 to 3", "3 to 4", "4 to 5"
		piesizes = [np.sum(self.steam_rate_2), np.sum(self.steam_rate_3), np.sum(self.steam_rate_4), np.sum(self.steam_rate_5)]
		piecolors = ["limegreen", "sandybrown", "hotpink", "grey"]
		plt.pie(piesizes, labels=pielabels, colors=piecolors, autopct="%1.1f%%", startangle=140)
		plt.axis("equal")
		plt.title("Percentage of Rating on Steam Games",size=20)
		plt.savefig("../result/Pie pct of Games with Rating_steam.png")
		plt.show()

		plt.figure(figsize=(8,6))
		piesizes = [np.sum(self.mobile_rate_2), np.sum(self.mobile_rate_3), np.sum(self.mobile_rate_4), np.sum(self.mobile_rate_5)]
		piecolors = ["limegreen", "sandybrown", "hotpink", "grey"]
		plt.pie(piesizes, labels=pielabels, colors=piecolors, autopct="%1.1f%%", startangle=140)
		plt.axis("equal")
		plt.title("Percentage of Rating on Mobile Games",size=20)
		plt.savefig("../result/Pie pct of Games with Rating_Mobile.png")
		plt.show()

	def plot_rating_Vs_price(self):
		sns.set()

		plt.figure(figsize=(8,6))
		Price_m1=sns.regplot(x=self.Data_mobile["Price"], y=self.Data_mobile["Average User Rating"],fit_reg=False)
		Price_m1.set_ylabel("Average User Rating",fontsize=16)
		Price_m1.set_title("Average User Rating vs. Price on mobile",fontsize=20)
		
		figpm1 = Price_m1.get_figure()
		figpm1.savefig("../result/Price_vs_rating_mobile.png")

		plt.figure(figsize=(8,6))
		Price_s1=sns.regplot(x=self.Data_steam["price"], y=self.Data_steam["Average User Rating"],fit_reg=False)
		Price_s1.set_title("Average User Rating vs. Price on steam",fontsize=20)
		Price_s1.set_ylabel("Average User Rating",fontsize=16)
		figps1 = Price_s1.get_figure()
		figps1.savefig("Price_vs_rating_steam.png")


	def plot_active_user_Vs_price_mobile(self):
		sns.set()

		Rating_cat=[]
		for i in self.Data_mobile.index:
			if 0<=self.Data_mobile["Average User Rating"][i]<=2:
				Rating_cat.append("2 or below")
			elif 2<self.Data_mobile["Average User Rating"][i]<=3:
				Rating_cat.append("2 to 3")
			elif 3<self.Data_mobile["Average User Rating"][i]<=4:
				Rating_cat.append("3 to 4")
			elif 4<self.Data_mobile["Average User Rating"][i]<=5:
				Rating_cat.append("4 to 5")
			else:
				Rating_cat.append(np.nan)
		self.Data_mobile["Rating Level"]=pd.Series(Rating_cat)   
		color_dict = dict({"2 or below":"grey","2 to 3":"tab:green","3 to 4": "tab:blue","4 to 5": "tab:red"})
		plt.figure(figsize=(8,6))
		Price_m2=sns.scatterplot(x="Price", y="User Rating Count",data=self.Data_mobile,hue="Rating Level",hue_order=["4 to 5","3 to 4","2 to 3","2 or below"],palette=color_dict,alpha=0.5)
		Price_m2.set_title("Active User num vs. Price on mobile",fontsize=20)
		Price_m2.set(yscale="log")
		Price_m2.set_ylabel("Active User Number",fontsize=16)
		figpm2 = Price_m2.get_figure()
		figpm2.savefig("../result/Price_vs_active user_mobile.png")


		Rating_cats=[]
		for i in self.Data_steam.index:
			if 0<=self.Data_steam["Average User Rating"][i]<=2:
				Rating_cats.append("2 or below")
			elif 2<self.Data_steam["Average User Rating"][i]<=3:
				Rating_cats.append("2 to 3")
			elif 3<self.Data_steam["Average User Rating"][i]<=4:
				Rating_cats.append("3 to 4")
			elif 4<self.Data_steam["Average User Rating"][i]<=5:
				Rating_cats.append("4 to 5")
			else:
				Rating_cats.append(np.nan)
		self.Data_steam["Rating Level"]=pd.Series(Rating_cats)	   
		plt.figure(figsize=(8,6))
		Price_s2=sns.scatterplot(x="price", y="User Rating Count",data=self.Data_steam,hue="Rating Level",hue_order=["4 to 5","3 to 4","2 to 3","2 or below"],palette=color_dict,alpha=0.5)
		Price_s2.set_title("Active User num vs. Price on steam",fontsize=20)
		Price_s2.set(yscale="log")
		Price_s2.set_ylabel("Active User Number",fontsize=16)
		figps2 = Price_s2.get_figure()
		figps2.savefig("../result/Price_vs_active user_steam.png")

	def prepare_data(self):
		Rating_mean_m=self.Data_mobile["Average User Rating"].mean() # =4.06
		Data_m_evaluated=self.Data_mobile[self.Data_mobile["Average User Rating"]>Rating_mean_m] #3723 rows × 13 columns
		#Data_m_evaluated.to_csv("Data_mobile_evaluated.csv") 

		#steam
		Rating_mean_s=self.Data_steam["Average User Rating"].mean() # =3.57
		Data_s_evaluated=self.Data_steam[self.Data_steam["Average User Rating"]>Rating_mean_s] #15544 rows × 18 columns
		##Organize data according to Genres
		#mobile
		Genres_mobile=pd.Series(self.Data_mobile["Genres"])
		Genres_m=Genres_mobile.str.split(", ")
		self.Data_mobile["Genres"]=Genres_m #set Genres into str list
		Data_m_expG=self.Data_mobile.explode("Genres") #explode data depend on Genres
		#Data_m_expG.shape is (57704, 11)
		Data_m_expG.to_csv("Data_mobile__explode genres.csv")
		#top50% user-evaluated:
		Data_m_expG_eval=Data_m_expG[Data_m_expG["Average User Rating"]>Rating_mean_m] #13135 rows × 13 columns
		Data_m_expG_eval=Data_m_expG_eval[Data_m_expG_eval.Genres!="Games"] #Remove "Games":9412 rows × 13 columns

		#steam
		Genres_steam=pd.Series(self.Data_steam["Genres"])
		Genres_s=Genres_steam.str.split(";")
		self.Data_steam["Genres"]=Genres_s #set genres into str list
		Data_s_expG=self.Data_steam.explode("Genres") #explode data depend on genres
		#Data_s_expG.shape is (76462, 16)
		#Data_s_expG.to_csv("Data_steam__explode genres.csv")
		#top50% user-evaluated:
		Data_s_expG_eval=Data_s_expG[Data_s_expG["Average User Rating"]>Rating_mean_s] #42268 rows × 18 columns		

		Genre_m_user=Data_m_expG[["Genres","User Rating Count"]].groupby("Genres").sum().sort_values(by="User Rating Count",ascending=False).astype(int).drop("Games",axis=0)
		Genre_m_perc=Genre_m_user["User Rating Count"]/(Genre_m_user["User Rating Count"].sum(axis=0))*100 #calculate percent in unit of %
		Genre_m_user["Count_percent"]=Genre_m_perc.map("{:,.2f}%".format).replace(0,"N/A") #format percentage
		CountG_mobile=Genres_m.explode().value_counts().to_frame("Games_mobile")
		dep_m_perc=CountG_mobile["Games_mobile"]/(CountG_mobile["Games_mobile"].sum(axis=0))*100 
		CountG_mobile["Percentage"]=dep_m_perc.map("{:,.2f}%".format) #format percentage
		Genre_m_user_eval=Data_m_expG_eval[["Genres","User Rating Count"]].groupby("Genres").sum().sort_values(by="User Rating Count",ascending=False).astype(int)
		Genre_m_perc_eval=Genre_m_user_eval["User Rating Count"]/(Genre_m_user_eval["User Rating Count"].sum(axis=0))*100 #calculate percent in unit of %
		Genre_m_user_eval["Count_percent"]=Genre_m_perc_eval.map("{:,.2f}%".format).replace(0,"N/A") #format percentage

		Genre_s_user=Data_s_expG[["Genres","User Rating Count"]].groupby("Genres").sum().sort_values(by="User Rating Count",ascending=False)
		Genre_s_perc=Genre_s_user["User Rating Count"]/(Genre_s_user["User Rating Count"].sum(axis=0))*100 #calculate percent in unit of %
		Genre_s_user["Count_percent"]=Genre_s_perc.map("{:,.2f}%".format) #format percentage
		CountG_steam=Genres_s.explode().value_counts().to_frame("Games_steam")
		dep_s_perc=CountG_steam["Games_steam"]/(CountG_steam["Games_steam"].sum(axis=0))*100 
		CountG_steam["Percentage"]=dep_s_perc.map("{:,.2f}%".format) #format percentage
		Genre_s_user_eval=Data_s_expG_eval[["Genres","User Rating Count"]].groupby("Genres").sum().sort_values(by="User Rating Count",ascending=False)
		Genre_s_perc_eval=Genre_s_user_eval["User Rating Count"]/(Genre_s_user_eval["User Rating Count"].sum(axis=0))*100 #calculate percent in unit of %
		Genre_s_user_eval["Count_percent"]=Genre_s_perc_eval.map("{:,.2f}%".format) #format percentage


		self.Genre_m_user = Genre_m_user
		self.Genre_m_user_eval = Genre_m_user_eval
		self.Genre_m_user_eval.rename(columns={"User Rating Count":"Evaluated Count","Count_percent":"Evaluated Percent"},inplace=True)
		self.CountG_mobile = CountG_mobile
		self.Genre_m_perc_eval = Genre_m_perc_eval
		self.Data_m_expG_eval = Data_m_expG_eval

		self.Genre_s_user = Genre_s_user
		self.Genre_s_user_eval = Genre_s_user_eval
		self.Genre_s_user_eval.rename(columns={"User Rating Count":"Evaluated Count","Count_percent":"Evaluated Percent"},inplace=True)
		self.CountG_steam = CountG_steam
		self.Genre_s_perc_eval = Genre_s_perc_eval
		self.Data_s_expG_eval = Data_s_expG_eval



	def plot_positive_count(self):
		Genre_m_comb=pd.concat([self.Genre_m_user, self.Genre_m_user_eval], axis=1,sort=True)
		plt.figure(figsize=(8,6))
		Genre_m3=sns.regplot(x=Genre_m_comb["User Rating Count"], y=Genre_m_comb["Evaluated Count"],fit_reg=True)
		Genre_m3.set_xlabel("Active User Count",fontsize=16)
		Genre_m3.set_ylabel("Positive-rated User Count",fontsize=16)
		Genre_m3.set_title("Mobile",fontsize=20)
		Genre_m3.set(yscale="log",xscale="log")
		Genre_m3.set_ylim([10,max(Genre_m_comb["Evaluated Count"])*2])
		Genre_m3.set_xlim([10,max(Genre_m_comb["User Rating Count"])*2])
		figpm3 = Genre_m3.get_figure()
		figpm3.savefig("../result/Total vs Evaluated_mobile.png")

		Genre_s_comb=pd.concat([self.Genre_s_user, self.Genre_s_user_eval], axis=1,sort=True)
		plt.figure(figsize=(8,6))
		Genre_s3=sns.regplot(x=Genre_s_comb["User Rating Count"], y=Genre_s_comb["Evaluated Count"],fit_reg=True)
		Genre_s3.set_xlabel("Active User Count",fontsize=16)
		Genre_s3.set_ylabel("Positive-rated User Count",fontsize=16)
		Genre_s3.set_title("Steam",fontsize=20)
		Genre_s3.set(yscale="log",xscale="log")
		Genre_s3.set_ylim([10,max(Genre_s_comb["Evaluated Count"])*2])
		Genre_s3.set_xlim([10,max(Genre_s_comb["User Rating Count"])*2])
		figps3 = Genre_s3.get_figure()
		figps3.savefig("../result/Total vs Evaluated_steam.png")

	def plot_developed_Vs_active_user(self):
		Game_m_comp=pd.concat([self.Genre_m_user, self.CountG_mobile], axis=1,sort=True)
		plt.figure(figsize=(8,6))
		Genre_m4=sns.regplot(x=Game_m_comp["User Rating Count"], y=Game_m_comp["Games_mobile"],fit_reg=False)
		Genre_m4.set_xlabel("Active User Count",fontsize=16)
		Genre_m4.set_ylabel("Games Developed",fontsize=16)
		Genre_m4.set_title("Mobile",fontsize=20)
		Genre_m4.set(yscale="log",xscale="log")
		Genre_m4.set_ylim([10,max(Game_m_comp["Games_mobile"])*2])
		Genre_m4.set_xlim([10,max(Game_m_comp["User Rating Count"])*2])
		figpm4 = Genre_m4.get_figure()
		figpm4.savefig("../result/User vs Developer_mobile.png")

		Game_s_comp=pd.concat([self.Genre_s_user, self.CountG_steam], axis=1,sort=True)
		plt.figure(figsize=(8,6))
		Genre_s4=sns.regplot(x=Game_s_comp["User Rating Count"], y=Game_s_comp["Games_steam"],fit_reg=False)	
		Genre_s4.set_xlabel("Active User Count",fontsize=16)
		Genre_s4.set_ylabel("Games Developed",fontsize=16)
		Genre_s4.set_title("Steam",fontsize=20)
		Genre_s4.set(yscale="log",xscale="log")
		Genre_s4.set_ylim([1,max(Game_s_comp["Games_steam"])*2])
		Genre_s4.set_xlim([1,max(Game_s_comp["User Rating Count"])*2])
		figps4 = Genre_s4.get_figure()
		figps4.savefig("../result/User vs Developer_steam.png")



	def plot_user_count_Vs_genres(self):
		plotm4=plt.figure(figsize=(8,6)) #plot Top10 bars
		plt.ylabel = "Top 10 Genres"
		plt.xlabel = "Pencent of User Rating count, %"
		s_plot=sns.barplot(y = self.Genre_m_user_eval.index[:10], x = self.Genre_m_perc_eval[:10], orient="h",palette="Greens_d")
		s_plot.set_ylabel("Top 10 Genres on Mobile",fontsize=16)
		s_plot.set_xlabel("Pencent of User Rating count, %",fontsize=16)
		s_plot.set_title("Percentage of Mobile User Count for Genres",fontsize=20)
		s_plot.tick_params(labelsize=12)
		plotm4.savefig("../result/Percentage of Mobile-User Count for Genres.png",bbox_inches="tight")
		plt.show()

		plots4=plt.figure(figsize=(8,6)) #plot Top10 bars
		plt.ylabel = "Top 10 Genres"
		plt.xlabel = "Pencent of User Rating count, %"
		s_plot=sns.barplot(y = self.Genre_s_user_eval.index[:10], x = self.Genre_s_perc_eval[:10], orient="h",palette="Blues_d")
		s_plot.set_ylabel("Top 10 Genres on Steam",fontsize=16)
		s_plot.set_xlabel("Pencent of User Rating count, %",fontsize=16)
		s_plot.set_title("Percentage of Steam User Count for Genres",fontsize=20)
		s_plot.tick_params(labelsize=12)
		plots4.savefig("../result/Percentage of Steam-User Count for Genres.png",bbox_inches="tight")
		plt.show()


	def generate_animation_for_genres(self):
		#mobile
	    ReleaseTime_mobile=self.Data_m_expG_eval.filter(["Genres","Release Year","Release Month","User Rating Count"],axis=1).groupby(["Release Year","Genres"])["User Rating Count"].sum().reset_index()
	    Mobile_Top10_ind=self.Genre_m_user_eval.index[:10]
	    ReleaseTime_m_top10=ReleaseTime_mobile[ReleaseTime_mobile["Genres"].isin(Mobile_Top10_ind)].reset_index()
	    Genres_cat_m=pd.Series(pd.Categorical(ReleaseTime_m_top10["Genres"],ordered=True,categories=Mobile_Top10_ind))
	    ReleaseTime_m_top10["Genres_cat"]=Genres_cat_m
	    
	    plt.figure()
	    figm, (axm1, axm2) = plt.subplots(1,2,figsize=(20,8))
	    figm.subplots_adjust(bottom=0.15, left=0.2)    
	    Top10dictm={i:[] for i in Mobile_Top10_ind} 
	    yearsm=[]
	    def Mobile_barplot_by_year(select_year):
	        yearsm.append(select_year)
	        if yearsm[-1]==yearsm[0]:
	            global Top10dictm
	            Top10dictm={i:[] for i in Mobile_Top10_ind}  
	               
	        select_countm=ReleaseTime_m_top10[ReleaseTime_m_top10["Release Year"].eq(select_year)].sort_values(by="Genres_cat",ascending=False)
	        
	        for key in Top10dictm:
	            if key in list(select_countm["Genres"]):
	                keyind=pd.Index(select_countm["Genres"]).get_loc(key)
	                Top10dictm[key]=Top10dictm[key]+[select_countm["User Rating Count"].iloc[keyind]]
	            else:
	                Top10dictm[key]=Top10dictm[key]+[0]
	            
	        Genrename=list(Top10dictm.keys())
	        Countvalue=np.array(list(Top10dictm.values()))
	        Count_cum=Countvalue.cumsum(axis=1)
	        #max num of color:10 #can be set differently
	        itcolor = it.count(start=0.2, step=0.08)
	        year_colors=plt.get_cmap("Greens")(list(next(itcolor) for _ in range(Countvalue.shape[1])))
	        #year_colors=plt.get_cmap("Blues")(np.linspace(0.15,0.85,Countvalue.shape[1]))
	        
	        axm1.clear()
	        axm1.invert_yaxis()
	        for i,(year,color) in enumerate(zip(yearsm,year_colors)):
	            widths=Countvalue[:,i]
	            starts=Count_cum[:,i]-widths
	            
	            axm1.barh(Genrename,widths,left=starts,label=year,color=color)
	            xcenters=starts+widths /2
	            axm1.text(1, 0.4, select_year, transform=axm1.transAxes, color="#777777", size=52, ha="right", weight=800)
	        
	        axm1.legend(ncol=len(yearsm), bbox_to_anchor=(0, 1),loc="lower left", fontsize="medium",handletextpad=0.1,handlelength=0.8,columnspacing=0.6)
	        axm1.set_xlabel("Cumulative Count")
	        axm1.set_ylabel("Top10 Genre")
	        
	        axm2.clear()
	        data=pd.DataFrame(Top10dictm)
	        data_perc=data.divide(data.sum(axis=1),axis=0)
	        l=len(list(Top10dictm.values())[0])
	        yrstr = [str(i)[-2:] for i in yearsm[:l]]
	        
	        axm2.stackplot(yrstr,[data_perc.iloc[:,9],data_perc.iloc[:,8],data_perc.iloc[:,7],data_perc.iloc[:,6],data_perc.iloc[:,5],data_perc.iloc[:,4],data_perc.iloc[:,3],data_perc.iloc[:,2],data_perc.iloc[:,1],data_perc.iloc[:,0]],labels=Genrename[::-1])
	        handles, labels = axm2.get_legend_handles_labels()
	        axm2.legend(handles[::-1], labels[::-1],ncol=5,bbox_to_anchor=(0, 1),loc="lower left", fontsize="medium",handletextpad=0.1,handlelength=0.8,columnspacing=0.6)
	        axm2.set_xlabel("Year")
	        axm2.set_ylabel("Percentage each Year")
	        return axm1,axm2
	    
	    def init():
	        #do nothing
	        pass
	    # ffmpegpath = "D:\\ffmpeg\\ffmpeg-20200311-36aaee2-win64-static\\bin\\ffmpeg.exe"
	    # matplotlib.rcParams["animation.ffmpeg_path"] = ffmpegpath
	    Writer = animation.writers["ffmpeg"]
	    writer = Writer(fps=1, metadata=dict(artist="Me"), bitrate=1800)
	    
	    animatorm = animation.FuncAnimation(figm, Mobile_barplot_by_year, frames=range(2010,2020),init_func=init,interval=1000)
	    HTML(animatorm.to_jshtml()) 
	    animatorm.save("../result/Mobile_by_year_sub.mp4", writer=writer)
	    
	    #steam
	    ReleaseTime_steam=self.Data_s_expG_eval.filter(["Genres","Release Year","Release Month","User Rating Count"],axis=1).explode("Genres").groupby(["Release Year","Genres"])["User Rating Count"].sum().reset_index()
	    Steam_Top10_ind=self.Genre_s_user_eval.index[:10]
	    ReleaseTime_s_top10=ReleaseTime_steam[ReleaseTime_steam["Genres"].isin(Steam_Top10_ind)].reset_index()
	    Genres_cat_s=pd.Series(pd.Categorical(ReleaseTime_s_top10["Genres"],ordered=True,categories=Steam_Top10_ind))
	    ReleaseTime_s_top10["Genres_cat"]=Genres_cat_s
	    
	    plt.figure()
	    #fig,ax=plt.subplots(figsize=(10,8))
	    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,8))
	    fig.subplots_adjust(bottom=0.15, left=0.2)    
	    Top10dict={i:[] for i in Steam_Top10_ind} 
	    years=[]
	    def Steam_barplot_by_year(select_year):
	        years.append(select_year)
	        if years[-1]==years[0]:
	            global Top10dict
	            Top10dict={i:[] for i in Steam_Top10_ind}  
	        select_count=ReleaseTime_s_top10[ReleaseTime_s_top10["Release Year"].eq(select_year)].sort_values(by="Genres_cat",ascending=False)
	        
	        for key in Top10dict:
	            if key in list(select_count["Genres"]):
	                keyind=pd.Index(select_count["Genres"]).get_loc(key)
	                Top10dict[key]=Top10dict[key]+[select_count["User Rating Count"].iloc[keyind]]
	            else:
	                Top10dict[key]=Top10dict[key]+[0]
	            
	        Genrename=list(Top10dict.keys())
	        Countvalue=np.array(list(Top10dict.values()))
	        Count_cum=Countvalue.cumsum(axis=1)
	        #max num of color:10 #can be set differently
	        itcolor = it.count(start=0.025, step=0.05)
	        year_colors=plt.get_cmap("tab20c")(list(next(itcolor) for _ in range(Countvalue.shape[1])))
	        #year_colors=plt.get_cmap("Blues")(np.linspace(0.15,0.85,Countvalue.shape[1]))
	        
	        ax1.clear()
	        ax1.invert_yaxis()
	        for i,(year,color) in enumerate(zip(years,year_colors)):
	            widths=Countvalue[:,i]
	            starts=Count_cum[:,i]-widths
	            
	            ax1.barh(Genrename,widths,left=starts,label=year,color=color)
	            xcenters=starts+widths /2
	            ax1.text(1, 0.4, select_year, transform=ax1.transAxes, color="#777777", size=52, ha="right", weight=800)
	        
	        ax1.legend(ncol=10, bbox_to_anchor=(0, 1),loc="lower left", fontsize="medium",handletextpad=0.1,handlelength=0.8,columnspacing=0.6)
	        ax1.set_xlabel("Cumulative Count")
	        ax1.set_ylabel("Top10 Genre")
	        
	        ax2.clear()
	        data=pd.DataFrame(Top10dict)
	        data_perc=data.divide(data.sum(axis=1),axis=0)
	        l=len(list(Top10dict.values())[0])
	        yrstr = [str(i)[-2:] for i in years[:l]]
	        ax2.stackplot(yrstr,[data_perc.iloc[:,9],data_perc.iloc[:,8],data_perc.iloc[:,7],data_perc.iloc[:,6],data_perc.iloc[:,5],data_perc.iloc[:,4],data_perc.iloc[:,3],data_perc.iloc[:,2],data_perc.iloc[:,1],data_perc.iloc[:,0]],labels=Genrename[::-1])
	        handles, labels = ax2.get_legend_handles_labels()
	        ax2.legend(handles[::-1], labels[::-1],ncol=5,bbox_to_anchor=(0, 1),loc="lower left", fontsize="medium",handletextpad=0.1,handlelength=0.8,columnspacing=0.6)
	        ax2.set_xlabel("Year")
	        ax2.set_ylabel("Percentage each Year")
	        return ax1,ax2

	    
	    animator = animation.FuncAnimation(fig, Steam_barplot_by_year, frames=range(2000,2020),init_func=init,interval=1000)
	    HTML(animator.to_jshtml()) 
	    animator.save("../result/Steam_by_year_sub.mp4", writer=writer)
	    plt.show()


