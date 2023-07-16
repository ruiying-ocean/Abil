
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
import matplotlib as mpl

# sns.set(rc={'figure.figsize':(11.7,8.27)})

# d = np.log(ds[["3-4 um", "5 um", "6-7 um", "8-12 um", "13-16 um"]].clip(min=0).mean(dim=["depth", "time", "lon"])).to_dataframe()
# d.reset_index(inplace=True)
# d = pd.melt(d, id_vars=["lat"], var_name="size class", value_name="biomass")

# g = sns.FacetGrid(d, col="size class",  hue="size class",  aspect= 0.2)
# g.map_dataframe(sns.lineplot, x="biomass", y="lat",  orient="y", size=50)


class plot_stats:

    def __init__(self, d, size_groups, regions):

        self.size_groups = size_groups
        self.regions = regions
        self.d = d
        dict = {species:k
            for k, v in self.regions.items()
            for species in v['provinces']}
        self.d['FID'] = self.d['FID'].map(dict) 
        self.d["total_log"] = np.log(self.d['total']+1)
        self.d = self.d.dropna()

    def niche_kernel_plot(self):
        print("")

    def logit_reg_plot(self, region, ax=None, x='total_log', y='simpson'):


        if ax==None:
            ax= plt.axes()

        if region!="Global":
            d = self.d[self.d["FID"] == region].copy()
        else:
            d = self.d.sample(100000).copy()

        ax.hist2d(x=d[x], y=d[y], bins=100, norm=mpl.colors.LogNorm())

        sns.regplot(data=d, y=y, x=x, ax=ax, logistic=True, n_boot=100, scatter=False).set(title=region)



    def reg_plot(self, region, ax=None, x='total_log', y='simpson'):


        if ax==None:
            ax= plt.axes()

        if region!="Global":
            d = self.d[self.d["FID"] == region].copy()
        else:
            d = self.d.sample(100000).copy()

        ax.hist2d(x=d[x], y=d[y], bins=100, norm=mpl.colors.LogNorm())
        ax.set(title=region)
    #    sns.regplot(data=d, y=y, x=x, ax=ax,  n_boot=100, scatter=False).set(title=region)
        ax.set_xlabel('biomass')
        ax.set_ylabel('diversity')



    def top_size_groups_boxplot(self, region, ax=None, log=False):
        if ax==None:
            ax= plt.axes()


        if region!="Global":
            d = self.d[self.d["FID"] == region]
        else:
            d = self.d.sample(100000).copy()



        dict = {species:k
            for k, v in self.size_groups.items()
            for species in v['species']}

        d = (d.rename(columns=dict)
            .groupby(level=0, axis=1, dropna=False)).sum( min_count=1)




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


        dict = {species:k
            for k, v in self.regions.items()
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


    # def make_robinson_plot(self, cocco_plot):

    #     projection = ccrs.Robinson(central_longitude=-160)
    #     ax= plt.axes(projection=projection)
    #     subplot_kws=dict(projection=projection,
    #                     facecolor='black')
    #     p = cocco_plot.plot(
    #                     x='lon', y='lat',
    #                     cmap=cm.viridis,
    #                     subplot_kws=subplot_kws,
    #                     transform=ccrs.PlateCarree(),
    #                     robust=True)
    #     ax.coastlines()

    #     ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k')
    #     ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    # def robinson_animation(self, cocco_plot):
    #     levels = range(12)
    #     fig, axes = plt.subplots()#ncols=1) #Creating the basis for the plot


    #     def animate(time):
    #         self.make_robinson_plot(self.cocco_plot.isel(time=time))

    #     ani = animation.FuncAnimation(fig, animate, 12, interval=400, blit=False)

    #     fig.suptitle("", fontsize= 18)
    #     ani.save('/home/phyto/animation.gif', writer='imagemagick', fps = 2) #Save animation as gif-file

    #     return(ani)



class plot_xy:

    def __init__(self, ds):
        self.ds = ds


    def latlon(self, variable, ax=None, fig=None, log=False, cmap=cm.viridis, vmin=None, vmax=None):

        if fig==None:
            fig = plt.figure(figsize=(20, 10))
        if ax==None:
            projection = ccrs.Robinson(central_longitude=-160)
            ax= plt.axes(projection=projection)
           
        cocco_plot = self.ds[variable].clip(min=0).mean(dim=["depth", "time"])
        if log==True:
            cocco_plot = np.log(cocco_plot+1)

        ax.coastlines()
        ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k', facecolor="gray")
        ax.gridlines(draw_labels=True,)

        p = cocco_plot.plot(
                        x='lon', y='lat',
                        cmap=cmap,
                        transform=ccrs.PlateCarree(),
                        robust=True,
                        add_colorbar=False,
                        vmin = vmin,
                        vmax = vmax,
                        ax = ax)
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])#
        ax.set_title(variable)
        plt.colorbar(p, cax=cax)



    def latlon_2(self, cocco_plot, ax=None, fig=None, log=False, cmap=cm.viridis, vmin=None, vmax=None):

        if fig==None:
            fig = plt.figure(figsize=(20, 10))
        if ax==None:
            projection = ccrs.Robinson(central_longitude=-160)
            ax= plt.axes(projection=projection)
           
        if log==True:
            cocco_plot = np.log(cocco_plot+1)

        ax.coastlines()
        ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k', facecolor="gray")
        ax.gridlines(draw_labels=True,)

        p = cocco_plot.plot(
                        x='lon', y='lat',
                        cmap=cmap,
                        transform=ccrs.PlateCarree(),
                        robust=True,
                        add_colorbar=False,
                        vmin = vmin,
                        vmax = vmax,
                        ax = ax)
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])#
        plt.colorbar(p, cax=cax)
        return(p)



    def robinson_animation(self, variable, ax=None, fig=None, log=False, cmap=cm.viridis, vmin=None, vmax=None):
        
        cocco_plot = self.ds[variable].clip(min=0).mean(dim=["depth"])

        levels = range(12)

        if fig==None:
            fig = plt.figure(figsize=(20, 10))
        if ax==None:
            projection = ccrs.Robinson(central_longitude=-160)
            ax= plt.axes(projection=projection)


        def animate(time):
            self.latlon_2(cocco_plot.isel(time=time), ax=ax, fig=fig, log=log, cmap=cmap, vmin=vmin, vmax=vmax)

        ani = animation.FuncAnimation(fig, animate, 12, interval=400, blit=False)

        fig.suptitle("", fontsize= 18)
        ani.save('/home/phyto/animation.gif', writer='imagemagick', fps = 2) #Save animation as gif-file

        return(ani)




    def latdepth(self, variable, ax=None, fig=None, log=False):

        if fig==None:
            fig = plt.figure(figsize=(20, 10))
        if ax==None:
            ax= plt.axes()

        cocco_plot = self.ds[variable].clip(min=0).mean(dim=["lon", "time"])
        if log==True:
            cocco_plot = np.log(cocco_plot+1)
        
        p = cocco_plot.plot(
                        x='lat', y='depth',
                        cmap=cm.viridis,
                        robust=True,
                        add_colorbar=True,
                        ax = ax)
        
        ax.set_title(variable)
        ax.invert_yaxis()


    def lattime(self, variable, ax=None, fig=None, log=False):

        if fig==None:
            fig = plt.figure(figsize=(20, 10))
        if ax==None:
            ax= plt.axes()
           

        cocco_plot = self.ds[variable].mean(dim=["lon", "depth"]) #.fillna(0)
        if log==True:
            cocco_plot = np.log(cocco_plot+1)
        
        
        p = cocco_plot.plot(
                        x='time', y='lat',
                        cmap=cm.viridis,
                        robust=True,
                        add_colorbar=True,
                        ax = ax)
        
        ax.set_title(variable)

    def training_latlon(self, d, variable, ax=None, fig=None):



        if fig==None:
            fig = plt.figure(figsize=(20, 10))
        if ax==None:
            projection = ccrs.Robinson(central_longitude=-160)
            ax= plt.axes(projection=projection)
           

        ax.coastlines()
        ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k', facecolor="gray")
        ax.gridlines(draw_labels=True,)


        plt.scatter(x=d['Longitude'], y=d['Latitude'],
                    color="dodgerblue",
                    s=50,
                    alpha=0.5,
                    transform=ccrs.PlateCarree()) ## Important

    #    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])#
        ax.set_title(variable)
    #    plt.colorbar(p, cax=cax)