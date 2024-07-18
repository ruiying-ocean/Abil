#%%
import cartopy.crs as ccrs
import cartopy as cart
from yaml import load, Loader
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors

#%%
with open('/home/mv23682/Documents/Abil/wiseman2024/ensemble_regressor.yml', 'r') as f:
    model_config = load(f, Loader=Loader)

date = '2024-05-06/'
run_name = '50/'
species = '2024-05-06_cp_ci50_obs'

df = pd.read_csv(model_config['local_root'] + model_config['path_out'] + date + run_name + species + ".csv")

try:
    df = df.rename({'level_0': 'time','level_1':'depth','level_2':'lat','level_3':'lon'})
except:
    pass

plt.rcParams.update({'font.size':20})
#%%
def pointmap(df, variable, ax=None, cmap = cm.viridis,
             fig=None, log=False, levels = None, title=None, units=None,
             add_colorbar=False, depth_integrated=False):

    #df = df[df['depth'] <= 20]

    # Plot points colored by the value of "calcification"
    calcification = df[variable]
    #calcification = calcification*(10**(-3)) # mmol C m^-3 d^-1
    if log:
        calcification = np.log10(calcification)  # Apply log transformation

    #define figure setup (subplots and dimensions)
    if fig==None:
        fig = plt.figure(figsize=(20, 10))
    if ax==None:
        projection = ccrs.Robinson(central_longitude=-160)
        ax= plt.axes(projection=projection)

    # setup map (coastlines and grid labels)
    ax.coastlines()
    ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k', facecolor="gray")
    ax.gridlines(draw_labels=True,)

    cmap = plt.get_cmap(cmap)
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    #create plot
    p = ax.scatter(df['lon'], df['lat'],
                   s=90,
                   c=calcification,
                   cmap=cmap,
                   norm=norm,
                   transform=ccrs.PlateCarree())


        #add title (variable name if undefined)
    if title==None:
        ax.set_title(variable, fontdict = {"fontsize": 24})
    else:
        ax.set_title(title, fontdict = {"fontsize": 24})
        if depth_integrated==True:
            cb = plt.colorbar(p,
                location = 'bottom',
                pad=0.08,
                shrink=0.7,
                extend='max'
                ).set_label(
                    label=units,
                    size=24)
        else:
            cb = plt.colorbar(p,
                location = 'bottom',
                pad=0.05,
                shrink=0.7,
                extend='max'
                ).set_label(
                    label=units,
                    size=24)
    ax.set_extent([20,21,-77,80])

    plt.show()

#%%
levels = np.arange(0,2.01,0.1)
pointmap(df, "Calcification", 
       levels=levels,
       log = True,
       title= r"Observed Calcium Carbonate Production (0-200m)",
       units=r"Log$_{10}$ Calcification (mmol C m$^{-3}$ d$^{-1}$)",
       cmap = cm.viridis
       )
plt.show()

#%%
levels = np.arange(-3,4.5,0.5)
pointmap(df, "Primary_Production", 
       levels=levels,
       log = True,
       title= r"Observed Primary Production (0-20m)",
       units=r"Log$_{10}$ Primary Production (mmol C m$^{-3}$ d$^{-1}$)",
       cmap = cm.jet
       )
plt.show()

# %%
