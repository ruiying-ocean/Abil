
import cartopy.crs as ccrs
import cartopy as cart
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import seaborn as sns
import pandas as pd
from matplotlib import scale as mscale


class plot:

    def __init__(self, ds):

        self.ds = ds

    def top_spp_boxplot(self, traits, region, log=False):
        fig = plt.figure(figsize=(20, 8))
        ax = fig.add_subplot(111)
        species = traits['species']
        d = self.ds.to_dataframe()
        d = d.sample(100000)
        d = d[d["FID"] == region]
        if log==True:
            d[species] = np.log1p(d[species])
        d.reset_index(inplace=True)
        d.set_index(['lat', 'lon', 'depth', 'time', 'FID'], inplace=True)
        df = d[species]
        df.reset_index(inplace=True)
        df = pd.melt(df, id_vars=['lat', 'lon', 'depth', 'time', 'FID'], value_vars=species)
        df_sum = df.groupby('variable').sum().sort_values(by='value',ascending=False)
        df_sum.reset_index(inplace=True)
        top_spp = df_sum['variable'].head(5)
        df_top = df.query("variable.isin(@top_spp).values")
        sns.boxplot(data = df_top, y="value", x="variable")

        return(plt)



    def make_robinson_plot(self, cocco_plot):

        projection = ccrs.Robinson(central_longitude=-160)
        ax= plt.axes(projection=projection)
        subplot_kws=dict(projection=projection,
                        facecolor='black')
        p = cocco_plot.plot(
                        x='lon', y='lat',
                        cmap=cm.viridis,
                        subplot_kws=subplot_kws,
                        transform=ccrs.PlateCarree(),
                        robust=True)
        ax.coastlines()

        ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k')
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())



    def make_plot(self, variable, log=False):
        fig = plt.figure(figsize=(20, 10))
        plt.title(variable)

        cocco_plot = self.ds[variable].clip(min=0).mean(dim=["depth", "time"])
        if log==True:
            cocco_plot = np.log(cocco_plot+1)

        projection = ccrs.Robinson(central_longitude=-160)
        ax= plt.axes(projection=projection)
        ax.coastlines()
        ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k', facecolor="gray")

        subplot_kws=dict(projection=projection,
                        facecolor='black')
        p = cocco_plot.plot(
                        x='lon', y='lat',
                        cmap=cm.viridis,
                        subplot_kws=subplot_kws,
                        transform=ccrs.PlateCarree(),
                        robust=True)
        




    def robinson_animation(self, cocco_plot):
        levels = range(12)
        fig, axes = plt.subplots()#ncols=1) #Creating the basis for the plot


        def animate(time):
            self.make_robinson_plot(self.cocco_plot.isel(time=time))

        ani = animation.FuncAnimation(fig, animate, 12, interval=400, blit=False)
        

        fig.suptitle("", fontsize= 18)
        ani.save('/home/phyto/animation.gif', writer='imagemagick', fps = 2) #Save animation as gif-file

        return(ani)
        