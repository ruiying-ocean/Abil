
import cartopy.crs as ccrs
import cartopy as cart
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import seaborn as sns
import pandas as pd
from matplotlib import scale as mscale
from mpl_toolkits.axes_grid1 import make_axes_locatable


from yaml import load
from yaml import CLoader as Loader
class plot:

    def __init__(self, ds):

        self.ds = ds


    def top_size_groups_boxplot(self, traits, region, ax, log=False):

        d = self.ds.to_dataframe()
        d = d.sample(100000)

        with open('/home/phyto/CoccoML/size_groups.yml', 'r') as f:
            size_groups = load(f, Loader=Loader)

        dict = {species:k
            for k, v in size_groups.items()
            for species in v['species']}

        d = (d.rename(columns=dict)
            .groupby(level=0, axis=1, dropna=False)).sum( min_count=1)

        with open('/home/phyto/CoccoML/regions.yml', 'r') as f:
            regions = load(f, Loader=Loader)

        dict = {species:k
            for k, v in regions.items()
            for species in v['provinces']}

        d['FID'] = d['FID'].map(dict) 

        if region!="Global":
            d = d[d["FID"] == region]

        sizes = ['3-4 um', '5 um', '6-7 um', '8-12 um', '13-16 um']


        if log==True:
            d[sizes] = np.log1p(d[sizes])

        d.reset_index(inplace=True)
        d.set_index(['lat', 'lon', 'depth', 'time', 'FID'], inplace=True)


        df = d[sizes] #subset size groups
        d.reset_index(inplace=True)
        df = pd.melt(d, id_vars=['lat', 'lon', 'depth', 'time', 'FID'], value_vars=sizes)
        sns.boxplot(data = df, y="value", x="variable", ax=ax).set(title=region, xlabel='', ylabel='log biomass')



    def top_spp_boxplot(self, traits, region, ax, log=False):
        species = traits['species']
        species_abrv = traits['species (abrv)']

        d = self.ds.to_dataframe()
        d = d.sample(100000)



        with open('/home/phyto/CoccoML/regions.yml', 'r') as f:
            regions = load(f, Loader=Loader)


        dict = {species:k
            for k, v in regions.items()
            for species in v['provinces']}

        d['FID'] = d['FID'].map(dict) 

        if region!="Global":
            d = d[d["FID"] == region]


        if log==True:
            d[species] = np.log1p(d[species])
        d.reset_index(inplace=True)
        d.set_index(['lat', 'lon', 'depth', 'time', 'FID'], inplace=True)
        df = d[species]

        test = pd.Series(traits['species (abrv)'].values, index=traits.species).to_dict()
        df.rename(columns=test, inplace=True)



        d = None
        df.reset_index(inplace=True)
        df = pd.melt(df, id_vars=['lat', 'lon', 'depth', 'time', 'FID'], value_vars=species_abrv)
        df_sum = df.groupby('variable').sum().sort_values(by='value',ascending=False)
        df_sum.reset_index(inplace=True)
        top_spp = df_sum['variable'].head(5)
        df_top = df.query("variable.isin(@top_spp).values")
        sns.boxplot(data = df_top, y="value", x="variable", ax=ax).set(title=region, xlabel='', ylabel='log biomass')

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
        plt.axis('off')

        cocco_plot = self.ds[variable].clip(min=0).mean(dim=["depth", "time"])
        if log==True:
            cocco_plot = np.log(cocco_plot+1)

        projection = ccrs.Robinson(central_longitude=-160)
        ax= plt.axes(projection=projection)
        ax.coastlines()
        ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k', facecolor="gray")
        ax.gridlines(draw_labels=True,)


        subplot_kws=dict(projection=projection,
                        facecolor='black')
        p = cocco_plot.plot(
                        x='lon', y='lat',
                        cmap=cm.viridis,
                        subplot_kws=subplot_kws,
                        transform=ccrs.PlateCarree(),
                        robust=True,
                        add_colorbar=False)
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])

        plt.colorbar(p, cax=cax)




    def robinson_animation(self, cocco_plot):
        levels = range(12)
        fig, axes = plt.subplots()#ncols=1) #Creating the basis for the plot


        def animate(time):
            self.make_robinson_plot(self.cocco_plot.isel(time=time))

        ani = animation.FuncAnimation(fig, animate, 12, interval=400, blit=False)

        fig.suptitle("", fontsize= 18)
        ani.save('/home/phyto/animation.gif', writer='imagemagick', fps = 2) #Save animation as gif-file

        return(ani)
        