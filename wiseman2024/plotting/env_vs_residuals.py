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
with open('/home/mv23682/Documents/Abil_Calcif/configuration/ensemble_regressor_cluster.yml', 'r') as f:
    model_config = load(f, Loader=Loader)
    
run_name = 'wo_MLP/wo_log'
species = '20_Mar_obs'
df = pd.read_csv(model_config['local_root'] + model_config['path_out'] + run_name + "/" + species + ".csv")

try:
    df = df.rename({'level_0': 'time','level_1':'depth','level_2':'lat','level_3':'lon'})
except:
    pass
predictors = model_config['predictors']


#%% Define scatter function
def scatter(df,x,y,
            ax=None, fig=None,
            title=None,axis1=None,axis2=None):
    X = df[x]
    Y = df[y]
    try:
        p = ax[axis1,axis2].scatter(X,Y,s=1)
    except:
        p = ax[axis1].scatter(X,Y,s=1)

    #add title (variable name if undefined)
    if title==None:
        try:
            ax[axis1,axis2].set_title(x, fontdict = {"fontsize": 12})
        except:
            ax[axis1].set_title(x, fontdict = {"fontsize": 12})
    else:
        try:
            ax[axis1,axis2].set_title(title, fontdict = {"fontsize": 8})
        except:
            ax[axis1].set_title(title, fontdict = {"fontsize": 8})

#%%
def pointmap(df, variable, ax=None, cmap = cm.viridis,
             fig=None, log=False, levels = None, title=None, units=None,
             add_colorbar=False, depth_integrated=False):

    df = df[df['depth'] <= 20]

    # Plot points colored by the value of "calcification"
    calcification = df[variable]
    calcification = calcification*(10**(-3)) # mmol C m^-3 d^-1
    if log:
        calcification = np.log(calcification)  # Apply log transformation

    #define figure setup (subplots and dimensions)
    if fig==None:
        fig = plt.figure(figsize=(20, 10))
    if ax==None:
        projection = ccrs.Robinson(central_longitude=0)
        ax= plt.axes(projection=projection)

    # setup map (coastlines and grid labels)
    ax.coastlines()
    ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k', facecolor="gray")
    ax.gridlines(draw_labels=True,)

    cmap = plt.get_cmap(cmap)
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    #create plot
    p = ax.scatter(
                    df['lon'], df['lat'],
                    s=90,
                    c=calcification,
                    cmap=cmap, norm=norm,
                    transform=ccrs.PlateCarree())

        #add title (variable name if undefined)
    if title==None:
#        ax.set_title(variable, fontdict = {"fontsize": 18})
        if depth_integrated==True:
            cb = plt.colorbar(p,
                location = 'bottom',
                pad=0.05,
                shrink=0.7,
                extend='both'
                ).set_label(
                    label=units,
                    size=16)
        else:
            cb = plt.colorbar(p,
                location = 'bottom',
                pad=0.05,
                shrink=0.7,
                extend='both'
                ).set_label(
                    label=units,
                    size=16)
    else:
        ax.set_title(title, fontdict = {"fontsize": 18})
        if depth_integrated==True:
            cb = plt.colorbar(p,
                location = 'bottom',
                pad=0.05,
                shrink=0.7,
                extend='both'
                ).set_label(
                    label=units,
                    size=16)
        else:
            cb = plt.colorbar(p,
                location = 'bottom',
                pad=0.05,
                shrink=0.7,
                extend='both'
                ).set_label(
                    label=units,
                    size=16)

    plt.show()



#%% Plot Calcification Residuals
fig, ax = plt.subplots(3, 5)
fig.suptitle('Calcification_resid')
fig.set_size_inches(10,6)
fig.tight_layout(h_pad=2)

k=0
for i in range(3):
    for j in range(5):
        try:
            scatter(df,predictors[k],'Calcification_resid',
                ax=ax,
                fig=fig,
                title=None,
                axis1=i,axis2=j)
        except:
            pass
        k=k+1

#%% Plot Primary Production Residuals
fig, ax = plt.subplots(3, 5)
fig.suptitle('Primary_Production_resid')
fig.set_size_inches(10,6)
fig.tight_layout(h_pad=2)

k=0
for i in range(3):
    for j in range(5):
        try:
            scatter(df,predictors[k],'Primary_Production_resid',
                ax=ax,
                fig=fig,
                title=None,
                axis1=i,axis2=j)
        except:
            pass
        k=k+1

# %%
levels = np.arange(-0.01,0.011,0.00125)
pointmap(df, "Calcification_resid", 
       levels=levels,
       log = False,
#       title= r"Calcium Carbonate Production Residual",
       units=r"Calcification residual (mmol C m$^{-3}$ d$^{-1}$)",
       cmap = cm.seismic
       )
plt.show()

#%%
levels = np.arange(-0.2,0.21,0.025)
pointmap(df, "Primary_Production_resid", 
       levels=levels,
       log = False,
#       title= r"Observed Primary Production (0-20m)",
       units=r"Primary Production residual (mmol C m$^{-3}$ d$^{-1}$)",
       cmap = cm.seismic
       )
plt.show()

# %%
