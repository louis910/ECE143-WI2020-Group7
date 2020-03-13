#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import six


def outputtable_from_Df(data, col_width=3.0, row_height=0.625, font_size=14,
                     rowLoc='Right',colLoc='center',header_color='steelblue', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1],header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    #Edit table:
    mpl_table = ax.table(cellText=data.values, bbox=bbox,rowLabels=data.index,rowLoc=rowLoc,colLoc=colLoc,colLabels=data.columns, **kwargs)
    
    #Add index name:
    mpl_table.add_cell(0, -1, col_width,row_height/(len(data.index)*2),text='Genres')
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    
    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])        
    
    return ax
    
#Edit filename:
#plot=outputtable_from_Df(tableDataFrame, col_width=3.0,row_height=0.625, font_size=14)
#plt.savefig('foo.png')


# In[160]:


import itertools as it 
import functools as func
import numpy as np
import random as rd
import pandas as pd
from pandas.plotting import table
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML
import seaborn as sns


def classproject():
    ##Import excel
    Data_mobile=pd.read_csv('mobile_modified.csv')
    newGenres=[]
    for row in range(len(Data_mobile['Price'])):
        if Data_mobile['Price'][row]==0:
            newGenres.append(Data_mobile['Genres'][row]+', Free to Play')
        else:
            newGenres.append(Data_mobile['Genres'][row])
    Data_mobile.drop(['Genres'], axis=1)
    Data_mobile['Genres']=pd.Series(newGenres)
    
    Data_steam=pd.read_csv('steam_modified.csv')
    Data_steam.rename(columns={'genres':'Genres','rating_count':'User Rating Count','release_date':'Original Release Date','average_ratings':'Average User Rating'},inplace=True)
    #Data_mobile.shape is (16286, 11)
    #Data_steam.shape is (27075, 16)
    
    def add_month_yr_mobile(x):
        '''Input:pd.DataFrame object
           Output:pd.DataFrame object
           table=pd.read_csv('fname')
        '''
        assert not x.empty
        assert isinstance(x,pd.DataFrame)
    
        timecol=x['Original Release Date']
        yr=[]
        mon=[]
        for item in timecol.iteritems():
            ind1=item[1].find('/')
            ind2=item[1].rfind('/')
            mm=item[1][ind1+1:ind2]
            mon.append(int(mm))
            yy=item[1][ind2+1:ind2+5]
            yr.append(int(yy))

        x['Release Year']=pd.Series(yr)
        x['Release Month']=pd.Series(mon)
        return x
   
    def add_month_yr_steam(x):
        '''Input:pd.DataFrame object
           Output:pd.DataFrame object
           table=pd.read_csv('fname')
        '''
        assert not x.empty
        assert isinstance(x,pd.DataFrame)
    
        timecol=x['Original Release Date']
        yr=[]
        mon=[]
        for item in timecol.iteritems():
            ind1=item[1].find('-')
            ind2=item[1].rfind('-')
            mm=item[1][ind1+1:ind2]
            mon.append(int(mm))
            yy=item[1][:ind1]
            yr.append(int(yy))

        x['Release Year']=pd.Series(yr)
        x['Release Month']=pd.Series(mon)
        return x
    sns.set()
    ##Add release year and release month to pd.DataFrame and plot developer-end stat
    #mobile
    Data_mobile=add_month_yr_mobile(Data_mobile)
    #Data_mobile.to_csv('Data_mobile__add release year and month.csv')
    Data_mobile['Counter yr/rating']=1
    Data_mobile_v=Data_mobile.groupby(['Release Year','Average User Rating'])['Counter yr/rating'].sum()
    #Analyze price-related stat
    rate_2=np.zeros(12)
    rate_3=np.zeros(12)
    rate_4=np.zeros(12)
    rate_5=np.zeros(12)
    yrs=list(range(2008,2020))
    for i in range(len(Data_mobile_v.index)):
        for yr in range(len(yrs)):
            if Data_mobile_v.index[i][0]==yrs[yr] and Data_mobile_v.index[i][1]<=2:
                rate_2[yr]=rate_2[yr]+Data_mobile_v[i]
            elif Data_mobile_v.index[i][0]==yrs[yr] and Data_mobile_v.index[i][1]<=3:
                rate_3[yr]=rate_3[yr]+Data_mobile_v[i]
            elif Data_mobile_v.index[i][0]==yrs[yr] and Data_mobile_v.index[i][1]<=4:
                rate_4[yr]=rate_4[yr]+Data_mobile_v[i]
            elif Data_mobile_v.index[i][0]==yrs[yr] and Data_mobile_v.index[i][1]<=5:
                rate_5[yr]=rate_5[yr]+Data_mobile_v[i]
    bardata_m=pd.DataFrame(index=yrs,data={'2 or below':rate_2,'2 to 3':rate_3,'3 to 4':rate_4,'4 to 5':rate_5})       
    plt.figure()
    bardata_m.plot(kind='barh',stacked=True,colormap='Accent')
    plt.title('Mobile',size=20)
    plt.savefig('Number of Games with Rating by Year_mobile.png')
    
    plt.figure()
    pielabels = '2 or below', '2 to 3', '3 to 4', '4 to 5'
    piesize = [np.sum(rate_2), np.sum(rate_3), np.sum(rate_4), np.sum(rate_5)]
    piecolors = ['limegreen', 'sandybrown', 'hotpink', 'grey']
    plt.pie(piesize, labels=pielabels, colors=piecolors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title("Percentage of Rating on Mobile Games",size=20)
    plt.savefig('Pie pct of Games with Rating_mobile.png')
    plt.show()
    
    #sns.set()
    plt.figure(figsize=(8,6))
    Price_m1=sns.regplot(x=Data_mobile["Price"], y=Data_mobile["Average User Rating"],fit_reg=False)
    Price_m1.set_ylabel("Average User Rating",fontsize=16)
    Price_m1.set_title('Average User Rating vs. Price on mobile',fontsize=20)

    figpm1 = Price_m1.get_figure()
    figpm1.savefig('Price_vs_rating_mobile.png')
    
    Rating_cat=[]
    for i in Data_mobile.index:
        if 0<=Data_mobile['Average User Rating'][i]<=2:
            Rating_cat.append('2 or below')
        elif 2<Data_mobile['Average User Rating'][i]<=3:
            Rating_cat.append('2 to 3')
        elif 3<Data_mobile['Average User Rating'][i]<=4:
            Rating_cat.append('3 to 4')
        elif 4<Data_mobile['Average User Rating'][i]<=5:
            Rating_cat.append('4 to 5')
        else:
            Rating_cat.append(np.nan)
    Data_mobile['Rating Level']=pd.Series(Rating_cat)   
    color_dict = dict({'2 or below':'grey','2 to 3':'tab:green','3 to 4': 'tab:blue','4 to 5': 'tab:red'})
    plt.figure(figsize=(8,6))
    Price_m2=sns.scatterplot(x="Price", y="User Rating Count",data=Data_mobile,hue='Rating Level',hue_order=['4 to 5','3 to 4','2 to 3','2 or below'],palette=color_dict,alpha=0.5)
    Price_m2.set_title('Active User num vs. Price on mobile',fontsize=20)
    Price_m2.set(yscale='log')
    Price_m2.set_ylabel("Active User Number",fontsize=16)
    figpm2 = Price_m2.get_figure()
    figpm2.savefig('Price_vs_active user_mobile.png')
    
    
    
    #steam
    Data_steam=add_month_yr_steam(Data_steam)
    #Data_steam.to_csv('Data_steam__add release year and month.csv')
    Data_steam['Counter yr/rating']=1
    Data_steam['Average User Rating']=Data_steam['Average User Rating'].round(3)
    Data_steam_v=Data_steam.groupby(['Release Year','Average User Rating'])['Counter yr/rating'].sum().reset_index()
    #Analyze price-related stat
    rate_2s=np.zeros(12)
    rate_3s=np.zeros(12)
    rate_4s=np.zeros(12)
    rate_5s=np.zeros(12)
    for i in range(len(Data_steam_v)):
        for yr in range(len(yrs)):
            if Data_steam_v['Release Year'][i]==yrs[yr] and Data_steam_v['Average User Rating'][i]<=2:
                rate_2s[yr]=rate_2s[yr]+Data_steam_v['Counter yr/rating'][i]
            elif Data_steam_v['Release Year'][i]==yrs[yr] and Data_steam_v['Average User Rating'][i]<=3:
                rate_3s[yr]=rate_3s[yr]+Data_steam_v['Counter yr/rating'][i]
            elif Data_steam_v['Release Year'][i]==yrs[yr] and Data_steam_v['Average User Rating'][i]<=4:
                rate_4s[yr]=rate_4s[yr]+Data_steam_v['Counter yr/rating'][i]
            elif Data_steam_v['Release Year'][i]==yrs[yr] and Data_steam_v['Average User Rating'][i]<=5:
                rate_5s[yr]=rate_5s[yr]+Data_steam_v['Counter yr/rating'][i]
    bardata_s=pd.DataFrame(index=yrs,data={'2 or below':rate_2s,'2 to 3':rate_3s,'3 to 4':rate_4s,'4 to 5':rate_5s})       
    plt.figure(figsize=(8,6))
    bardata_s.plot(kind='barh',stacked=True,colormap='Accent')
    plt.title('Steam',size=20)
    plt.savefig('Number of Games with Rating by Year_steam.png')
    plt.show()
    
    plt.figure(figsize=(8,6))
    piesizes = [np.sum(rate_2s), np.sum(rate_3s), np.sum(rate_4s), np.sum(rate_5s)]
    piecolors = ['limegreen', 'sandybrown', 'hotpink', 'grey']
    plt.pie(piesizes, labels=pielabels, colors=piecolors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title("Percentage of Rating on Steam Games",size=20)
    plt.savefig('Pie pct of Games with Rating_steam.png')
    plt.show()
    
    plt.figure(figsize=(8,6))
    Price_s1=sns.regplot(x=Data_steam["price"], y=Data_steam["Average User Rating"],fit_reg=False)
    Price_s1.set_title('Average User Rating vs. Price on steam',fontsize=20)
    Price_s1.set_ylabel("Average User Rating",fontsize=16)
    figps1 = Price_s1.get_figure()
    figps1.savefig('Price_vs_rating_steam.png')
    
    Rating_cats=[]
    for i in Data_steam.index:
        if 0<=Data_steam['Average User Rating'][i]<=2:
            Rating_cats.append('2 or below')
        elif 2<Data_steam['Average User Rating'][i]<=3:
            Rating_cats.append('2 to 3')
        elif 3<Data_steam['Average User Rating'][i]<=4:
            Rating_cats.append('3 to 4')
        elif 4<Data_steam['Average User Rating'][i]<=5:
            Rating_cats.append('4 to 5')
        else:
            Rating_cats.append(np.nan)
    Data_steam['Rating Level']=pd.Series(Rating_cats)       
    plt.figure(figsize=(8,6))
    Price_s2=sns.scatterplot(x="price", y="User Rating Count",data=Data_steam,hue='Rating Level',hue_order=['4 to 5','3 to 4','2 to 3','2 or below'],palette=color_dict,alpha=0.5)
    Price_s2.set_title('Active User num vs. Price on steam',fontsize=20)
    Price_s2.set(yscale='log')
    Price_s2.set_ylabel("Active User Number",fontsize=16)
    figps2 = Price_s2.get_figure()
    figps2.savefig('Price_vs_active user_steam.png')
    
    
    ##Find top50% user-evaluated Games
    #mobile 
    Rating_mean_m=Data_mobile['Average User Rating'].mean() # =4.06
    Data_m_evaluated=Data_mobile[Data_mobile['Average User Rating']>Rating_mean_m] #3723 rows × 13 columns
    #Data_m_evaluated.to_csv('Data_mobile_evaluated.csv') 
    
    #steam
    Rating_mean_s=Data_steam['Average User Rating'].mean() # =3.57
    Data_s_evaluated=Data_steam[Data_steam['Average User Rating']>Rating_mean_s] #15544 rows × 18 columns
    #Data_s_evaluated.to_csv('Data_steam_evaluated.csv')
    
    ##Organize data according to Genres
    #mobile
    Genres_mobile=pd.Series(Data_mobile['Genres'])
    Genres_m=Genres_mobile.str.split(', ')
    Data_mobile['Genres']=Genres_m #set Genres into str list
    Data_m_expG=Data_mobile.explode('Genres') #explode data depend on Genres
    #Data_m_expG.shape is (57704, 11)
    Data_m_expG.to_csv('Data_mobile__explode genres.csv')
    #top50% user-evaluated:
    Data_m_expG_eval=Data_m_expG[Data_m_expG['Average User Rating']>Rating_mean_m] #13135 rows × 13 columns
    Data_m_expG_eval=Data_m_expG_eval[Data_m_expG_eval.Genres!='Games'] #Remove 'Games':9412 rows × 13 columns
    
    #steam
    Genres_steam=pd.Series(Data_steam['Genres'])
    Genres_s=Genres_steam.str.split(';')
    Data_steam['Genres']=Genres_s #set genres into str list
    Data_s_expG=Data_steam.explode('Genres') #explode data depend on genres
    #Data_s_expG.shape is (76462, 16)
    #Data_s_expG.to_csv('Data_steam__explode genres.csv')
    #top50% user-evaluated:
    Data_s_expG_eval=Data_s_expG[Data_s_expG['Average User Rating']>Rating_mean_s] #42268 rows × 18 columns
    

    ##Developer-End Data Analysis: Count num of Games for each Genre
    #mobile:drop "Games" #need to check
    CountG_mobile=Genres_m.explode().value_counts().to_frame('Games_mobile').drop('Games',axis=0)
    dep_m_perc=CountG_mobile['Games_mobile']/(CountG_mobile['Games_mobile'].sum(axis=0))*100 
    CountG_mobile['Percentage']=dep_m_perc.map('{:,.2f}%'.format) #format percentage
    plotm1=outputtable_from_Df(CountG_mobile.head(10), col_width=3.0,row_height=0.625, font_size=14)
    plt.savefig('Top10 Count num of Games for each Genre_mobile.png',bbox_inches='tight')
    
    #steam
    CountG_steam=Genres_s.explode().value_counts().to_frame('Games_steam')
    dep_s_perc=CountG_steam['Games_steam']/(CountG_steam['Games_steam'].sum(axis=0))*100 
    CountG_steam['Percentage']=dep_s_perc.map('{:,.2f}%'.format) #format percentage
    plots1=outputtable_from_Df(CountG_steam.head(10), col_width=3.0,row_height=0.625, font_size=14)
    plt.savefig('Top10 Count num of Games for each Genre_steam.png',bbox_inches='tight')
    
    ##User-End Data Analysis: 
    #Count total num of users who evaluated for each genre:
    #mobile
    Genre_m_user=Data_m_expG[['Genres','User Rating Count']].groupby('Genres').sum().sort_values(by='User Rating Count',ascending=False).astype(int).drop('Games',axis=0)
    Genre_m_perc=Genre_m_user['User Rating Count']/(Genre_m_user['User Rating Count'].sum(axis=0))*100 #calculate percent in unit of %
    Genre_m_user['Count_percent']=Genre_m_perc.map('{:,.2f}%'.format).replace(0,'N/A') #format percentage
    plotm2=outputtable_from_Df(Genre_m_user.head(10), col_width=3.0,row_height=0.625, font_size=14)
    plt.savefig('Top10 Count num of users evaluated for each Genre_mobile.png',bbox_inches='tight')
    #Top50% evaluated:
    Genre_m_user_eval=Data_m_expG_eval[['Genres','User Rating Count']].groupby('Genres').sum().sort_values(by='User Rating Count',ascending=False).astype(int)
    Genre_m_perc_eval=Genre_m_user_eval['User Rating Count']/(Genre_m_user_eval['User Rating Count'].sum(axis=0))*100 #calculate percent in unit of %
    Genre_m_user_eval['Count_percent']=Genre_m_perc_eval.map('{:,.2f}%'.format).replace(0,'N/A') #format percentage
    plotm3=outputtable_from_Df(Genre_m_user_eval.head(10), col_width=3.0,row_height=0.625, font_size=14)
    plt.savefig('Top50% evaluated count for each Genre_mobile.png',bbox_inches='tight')
    ######add more data analysis
    Genre_m_user_eval.rename(columns={'User Rating Count':'Evaluated Count','Count_percent':'Evaluated Percent'},inplace=True)
    Genre_m_comb=pd.concat([Genre_m_user, Genre_m_user_eval], axis=1,sort=True)
    plotm3=outputtable_from_Df(Genre_m_comb, col_width=3.0,row_height=0.625, font_size=14)
    plt.savefig('Table of Combined Genre data_mobile.png',bbox_inches='tight')
    plt.figure(figsize=(8,6))
    Genre_m3=sns.regplot(x=Genre_m_comb["User Rating Count"], y=Genre_m_comb["Evaluated Count"],fit_reg=True)
    Genre_m3.set_xlabel("Active User Count",fontsize=16)
    Genre_m3.set_ylabel("Positive-rated User Count",fontsize=16)
    Genre_m3.set_title('Mobile',fontsize=20)
    Genre_m3.set(yscale='log',xscale='log')
    Genre_m3.set_ylim([10,max(Genre_m_comb["Evaluated Count"])*2])
    Genre_m3.set_xlim([10,max(Genre_m_comb["User Rating Count"])*2])
    figpm3 = Genre_m3.get_figure()
    figpm3.savefig('Total vs Evaluated_mobile.png')
    
    Game_m_comp=pd.concat([Genre_m_user, CountG_mobile], axis=1,sort=True)
    plotm3=outputtable_from_Df(Game_m_comp, col_width=3.0,row_height=0.625, font_size=14)
    plt.savefig('Table of comparison_mobile.png',bbox_inches='tight')
    plt.figure(figsize=(8,6))
    Genre_m4=sns.regplot(x=Game_m_comp["User Rating Count"], y=Game_m_comp["Games_mobile"],fit_reg=False)
    Genre_m4.set_xlabel("Active User Count",fontsize=16)
    Genre_m4.set_ylabel("Games Developed",fontsize=16)
    Genre_m4.set_title('Mobile',fontsize=20)
    Genre_m4.set(yscale='log',xscale='log')
    Genre_m4.set_ylim([10,max(Game_m_comp["Games_mobile"])*2])
    Genre_m4.set_xlim([10,max(Game_m_comp["User Rating Count"])*2])
    figpm4 = Genre_m4.get_figure()
    figpm4.savefig('User vs Developer_mobile.png')
    
    #Bar plot for Top50% in percentage:
    plotm4=plt.figure(figsize=(8,6)) #plot Top10 bars
    plt.ylabel = "Top 10 Genres"
    plt.xlabel = "Pencent of User Rating count, %"
    s_plot=sns.barplot(y = Genre_m_user_eval.index[:10], x = Genre_m_perc_eval[:10], orient='h',palette="Greens_d")
    s_plot.set_ylabel("Top 10 Genres on Mobile",fontsize=16)
    s_plot.set_xlabel("Pencent of User Rating count, %",fontsize=16)
    s_plot.set_title("Percentage of Mobile User Count for Genres",fontsize=20)
    s_plot.tick_params(labelsize=12)
    plotm4.savefig("Percentage of Mobile-User Count for Genres.png",bbox_inches='tight')
    plt.show()
    
    #steam
    Genre_s_user=Data_s_expG[['Genres','User Rating Count']].groupby('Genres').sum().sort_values(by='User Rating Count',ascending=False)
    Genre_s_perc=Genre_s_user['User Rating Count']/(Genre_s_user['User Rating Count'].sum(axis=0))*100 #calculate percent in unit of %
    Genre_s_user['Count_percent']=Genre_s_perc.map('{:,.2f}%'.format) #format percentage
    plots2=outputtable_from_Df(Genre_s_user.head(10), col_width=3.0,row_height=0.625, font_size=14)
    plt.savefig('Top10 Count num of users evaluated for each Genre_steam.png',bbox_inches='tight')
    #Top50% evaluated:
    Genre_s_user_eval=Data_s_expG_eval[['Genres','User Rating Count']].groupby('Genres').sum().sort_values(by='User Rating Count',ascending=False)
    Genre_s_perc_eval=Genre_s_user_eval['User Rating Count']/(Genre_s_user_eval['User Rating Count'].sum(axis=0))*100 #calculate percent in unit of %
    Genre_s_user_eval['Count_percent']=Genre_s_perc_eval.map('{:,.2f}%'.format) #format percentage
    plots3=outputtable_from_Df(Genre_s_user_eval.head(10), col_width=3.0,row_height=0.625, font_size=14)
    plt.savefig('Top50% evaluated count for each Genre_steam.png',bbox_inches='tight')
    #Bar plot for Top50% in percentage:
    plots4=plt.figure(figsize=(8,6)) #plot Top10 bars
    plt.ylabel = "Top 10 Genres"
    plt.xlabel = "Pencent of User Rating count, %"
    s_plot=sns.barplot(y = Genre_s_user_eval.index[:10], x = Genre_s_perc_eval[:10], orient='h',palette="Blues_d")
    s_plot.set_ylabel("Top 10 Genres on Steam",fontsize=16)
    s_plot.set_xlabel("Pencent of User Rating count, %",fontsize=16)
    s_plot.set_title("Percentage of Steam User Count for Genres",fontsize=20)
    s_plot.tick_params(labelsize=12)
    plots4.savefig("Percentage of Steam-User Count for Genres.png",bbox_inches='tight')
    plt.show()
     
    ######add more data analysis
    Genre_s_user_eval.rename(columns={'User Rating Count':'Evaluated Count','Count_percent':'Evaluated Percent'},inplace=True)
    Genre_s_comb=pd.concat([Genre_s_user, Genre_s_user_eval], axis=1,sort=True)
    plots3=outputtable_from_Df(Genre_s_comb, col_width=3.0,row_height=0.625, font_size=14)
    plt.savefig('Table of Combined Genre data_steam.png',bbox_inches='tight')
    plt.figure(figsize=(8,6))
    Genre_s3=sns.regplot(x=Genre_s_comb["User Rating Count"], y=Genre_s_comb["Evaluated Count"],fit_reg=True)
    Genre_s3.set_xlabel("Active User Count",fontsize=16)
    Genre_s3.set_ylabel("Positive-rated User Count",fontsize=16)
    Genre_s3.set_title('Steam',fontsize=20)
    Genre_s3.set(yscale='log',xscale='log')
    Genre_s3.set_ylim([10,max(Genre_s_comb["Evaluated Count"])*2])
    Genre_s3.set_xlim([10,max(Genre_s_comb["User Rating Count"])*2])
    figps3 = Genre_s3.get_figure()
    figps3.savefig('Total vs Evaluated_steam.png')
    
    Game_s_comp=pd.concat([Genre_s_user, CountG_steam], axis=1,sort=True)
    plots3=outputtable_from_Df(Game_s_comp, col_width=3.0,row_height=0.625, font_size=14)
    plt.savefig('Table of comparison_mobile.png',bbox_inches='tight')
    plt.figure(figsize=(8,6))
    Genre_s4=sns.regplot(x=Game_s_comp["User Rating Count"], y=Game_s_comp["Games_steam"],fit_reg=False)    
    Genre_s4.set_xlabel("Active User Count",fontsize=16)
    Genre_s4.set_ylabel("Games Developed",fontsize=16)
    Genre_s4.set_title('Steam',fontsize=20)
    Genre_s4.set(yscale='log',xscale='log')
    Genre_s4.set_ylim([1,max(Game_s_comp["Games_steam"])*2])
    Genre_s4.set_xlim([1,max(Game_s_comp["User Rating Count"])*2])
    figps4 = Genre_s4.get_figure()
    figps4.savefig('User vs Developer_steam.png')
    
    ##Create animation for Top50% Count for Genres by years
    #mobile
    ReleaseTime_mobile=Data_m_expG_eval.filter(['Genres','Release Year','Release Month','User Rating Count'],axis=1).groupby(['Release Year','Genres'])['User Rating Count'].sum().reset_index()
    Mobile_Top10_ind=Genre_m_user_eval.index[:10]
    ReleaseTime_m_top10=ReleaseTime_mobile[ReleaseTime_mobile['Genres'].isin(Mobile_Top10_ind)].reset_index()
    Genres_cat_m=pd.Series(pd.Categorical(ReleaseTime_m_top10['Genres'],ordered=True,categories=Mobile_Top10_ind))
    ReleaseTime_m_top10['Genres_cat']=Genres_cat_m
    
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
               
        select_countm=ReleaseTime_m_top10[ReleaseTime_m_top10['Release Year'].eq(select_year)].sort_values(by='Genres_cat',ascending=False)
        
        for key in Top10dictm:
            if key in list(select_countm['Genres']):
                keyind=pd.Index(select_countm['Genres']).get_loc(key)
                Top10dictm[key]=Top10dictm[key]+[select_countm['User Rating Count'].iloc[keyind]]
            else:
                Top10dictm[key]=Top10dictm[key]+[0]
            
        Genrename=list(Top10dictm.keys())
        Countvalue=np.array(list(Top10dictm.values()))
        Count_cum=Countvalue.cumsum(axis=1)
        #max num of color:10 #can be set differently
        itcolor = it.count(start=0.2, step=0.08)
        year_colors=plt.get_cmap('Greens')(list(next(itcolor) for _ in range(Countvalue.shape[1])))
        #year_colors=plt.get_cmap('Blues')(np.linspace(0.15,0.85,Countvalue.shape[1]))
        
        axm1.clear()
        axm1.invert_yaxis()
        for i,(year,color) in enumerate(zip(yearsm,year_colors)):
            widths=Countvalue[:,i]
            starts=Count_cum[:,i]-widths
            
            axm1.barh(Genrename,widths,left=starts,label=year,color=color)
            xcenters=starts+widths /2
            axm1.text(1, 0.4, select_year, transform=axm1.transAxes, color='#777777', size=52, ha='right', weight=800)
        
        axm1.legend(ncol=len(yearsm), bbox_to_anchor=(0, 1),loc='lower left', fontsize='medium',handletextpad=0.1,handlelength=0.8,columnspacing=0.6)
        axm1.set_xlabel('Cumulative Count')
        axm1.set_ylabel('Top10 Genre')
        
        axm2.clear()
        data=pd.DataFrame(Top10dictm)
        data_perc=data.divide(data.sum(axis=1),axis=0)
        l=len(list(Top10dictm.values())[0])
        yrstr = [str(i)[-2:] for i in yearsm[:l]]
        
        axm2.stackplot(yrstr,[data_perc.iloc[:,9],data_perc.iloc[:,8],data_perc.iloc[:,7],data_perc.iloc[:,6],data_perc.iloc[:,5],data_perc.iloc[:,4],data_perc.iloc[:,3],data_perc.iloc[:,2],data_perc.iloc[:,1],data_perc.iloc[:,0]],labels=Genrename[::-1])
        handles, labels = axm2.get_legend_handles_labels()
        axm2.legend(handles[::-1], labels[::-1],ncol=5,bbox_to_anchor=(0, 1),loc='lower left', fontsize='medium',handletextpad=0.1,handlelength=0.8,columnspacing=0.6)
        axm2.set_xlabel('Year')
        axm2.set_ylabel('Percentage each Year')
        return axm1,axm2
    
    def init():
        #do nothing
        pass
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)
    
    animatorm = animation.FuncAnimation(figm, Mobile_barplot_by_year, frames=range(2010,2020),init_func=init,interval=1000)
    HTML(animatorm.to_jshtml()) 
    animatorm.save('Mobile_by_year_sub.mp4', writer=writer)
    
    #steam
    ReleaseTime_steam=Data_s_expG_eval.filter(['Genres','Release Year','Release Month','User Rating Count'],axis=1).explode('Genres').groupby(['Release Year','Genres'])['User Rating Count'].sum().reset_index()
    Steam_Top10_ind=Genre_s_user_eval.index[:10]
    ReleaseTime_s_top10=ReleaseTime_steam[ReleaseTime_steam['Genres'].isin(Steam_Top10_ind)].reset_index()
    Genres_cat_s=pd.Series(pd.Categorical(ReleaseTime_s_top10['Genres'],ordered=True,categories=Steam_Top10_ind))
    ReleaseTime_s_top10['Genres_cat']=Genres_cat_s
    
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
        select_count=ReleaseTime_s_top10[ReleaseTime_s_top10['Release Year'].eq(select_year)].sort_values(by='Genres_cat',ascending=False)
        
        for key in Top10dict:
            if key in list(select_count['Genres']):
                keyind=pd.Index(select_count['Genres']).get_loc(key)
                Top10dict[key]=Top10dict[key]+[select_count['User Rating Count'].iloc[keyind]]
            else:
                Top10dict[key]=Top10dict[key]+[0]
            
        Genrename=list(Top10dict.keys())
        Countvalue=np.array(list(Top10dict.values()))
        Count_cum=Countvalue.cumsum(axis=1)
        #max num of color:10 #can be set differently
        itcolor = it.count(start=0.025, step=0.05)
        year_colors=plt.get_cmap('tab20c')(list(next(itcolor) for _ in range(Countvalue.shape[1])))
        #year_colors=plt.get_cmap('Blues')(np.linspace(0.15,0.85,Countvalue.shape[1]))
        
        ax1.clear()
        ax1.invert_yaxis()
        for i,(year,color) in enumerate(zip(years,year_colors)):
            widths=Countvalue[:,i]
            starts=Count_cum[:,i]-widths
            
            ax1.barh(Genrename,widths,left=starts,label=year,color=color)
            xcenters=starts+widths /2
            ax1.text(1, 0.4, select_year, transform=ax1.transAxes, color='#777777', size=52, ha='right', weight=800)
        
        ax1.legend(ncol=10, bbox_to_anchor=(0, 1),loc='lower left', fontsize='medium',handletextpad=0.1,handlelength=0.8,columnspacing=0.6)
        ax1.set_xlabel('Cumulative Count')
        ax1.set_ylabel('Top10 Genre')
        
        ax2.clear()
        data=pd.DataFrame(Top10dict)
        data_perc=data.divide(data.sum(axis=1),axis=0)
        l=len(list(Top10dict.values())[0])
        yrstr = [str(i)[-2:] for i in years[:l]]
        ax2.stackplot(yrstr,[data_perc.iloc[:,9],data_perc.iloc[:,8],data_perc.iloc[:,7],data_perc.iloc[:,6],data_perc.iloc[:,5],data_perc.iloc[:,4],data_perc.iloc[:,3],data_perc.iloc[:,2],data_perc.iloc[:,1],data_perc.iloc[:,0]],labels=Genrename[::-1])
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(handles[::-1], labels[::-1],ncol=5,bbox_to_anchor=(0, 1),loc='lower left', fontsize='medium',handletextpad=0.1,handlelength=0.8,columnspacing=0.6)
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Percentage each Year')
        return ax1,ax2

    
    animator = animation.FuncAnimation(fig, Steam_barplot_by_year, frames=range(2000,2020),init_func=init,interval=1000)
    HTML(animator.to_jshtml()) 
    animator.save('Steam_by_year_sub.mp4', writer=writer)
    plt.show()


# In[161]:


classproject()


# In[ ]:




