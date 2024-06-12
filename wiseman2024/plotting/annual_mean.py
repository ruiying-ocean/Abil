#%%
import cartopy.crs as ccrs
import cartopy as cart
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import xarray as xr
from yaml import load, Loader
from matplotlib import cm
from matplotlib.ticker import LogFormatter

#%%
with open('/home/mv23682/Documents/Abil/wiseman2024/ensemble_regressor.yml', 'r') as f:
    model_config = load(f, Loader=Loader)

run_name = '50'
species = '2024-05-06_cp_ci50'
ds = xr.open_dataset(model_config['local_root'] + model_config['path_out'] + run_name + "/" + species + ".nc")
try:
    ds = ds.rename({'level_0': 'time','level_1':'depth','level_2':'lat','level_3':'lon'})
except:
    pass
# %%
class plot_xy:
    def __init__(self, ds):
        self.ds = ds

    def latlon(self, variable, 
            ax=None, 
            fig=None, 
            add_contour=True, 
            log=False, 
            add_colorbar=True, 
            cmap=cm.viridis, 
            vmin=None, 
            vmax=None,
            start_depth=None, 
            end_depth=None, 
            depth_integrated=False,
            title=None,
            norm=None):

        #subset the data so it can be plotted:
        depth_range = slice(start_depth, end_depth)
        lons = np.arange(-180, 180, 1)
        if depth_integrated==True:
            cocco_plot = ds[variable].sel(depth=depth_range).sum(dim=["depth"]).mean(dim=["time"])
            cocco_plot = cocco_plot*(10**(-3))*12.01*5 # mg C m^-2 d^-1
        else:
            cocco_plot = ds[variable].sel(depth=depth_range).mean(dim=["depth","time"])
            cocco_plot = cocco_plot#*(10**(-3)) # mmol C m^-3 d^-1


        #define figure setup (subplots and dimensions)
        if fig==None:
            fig = plt.figure(figsize=(20, 10))
        if ax==None:
            projection = ccrs.Robinson(central_longitude=-160)
            ax= plt.axes(projection=projection)

        #apply log transformation if desired:
        #if log==True:
        #    cocco_plot = np.log10(cocco_plot)
        #else:
        cocco_plot = cocco_plot.where(cocco_plot != 0, float('nan'))

        # setup map (coastlines and grid labels)
        ax.coastlines()
        ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k', facecolor="gray")
        ax.gridlines(draw_labels=True,)

        #create plot
        p = cocco_plot.plot(
                        x='lon', y='lat',
                        cmap=cmap,
                        transform=ccrs.PlateCarree(),
                        robust=True,
                        add_colorbar=False,
                        vmin = vmin,
                        vmax = vmax,
                        ax = ax,
                        norm=norm)

        #add some contours:
        if add_contour == True:
            cocco_plot.plot.contour(x='lon', y='lat',
                            ax = ax)

        #add title (variable name if undefined)
        if title==None:
            ax.set_title(variable, fontdict = {"fontsize": 24})
        else:
            ax.set_title(title, fontdict = {"fontsize": 24})
            if log==True:
                formatterInt = LogFormatter(10, labelOnlyBase=False)
                formatter = LogFormatter(10, labelOnlyBase=False, minor_thresholds=(2,0.1))
                if depth_integrated==True:
                    cb = plt.colorbar(p,
                        location = 'bottom',
                        pad=0.08,
                        shrink=0.7,
                        extend='both',
                        format=formatterInt,
                        ticks=[1,5,10,50,100]
                        ).set_label(
                            label="Calcification (mg C m$^{-2}$ d$^{-1}$)",
                            size=24)
                else:
                    cb = plt.colorbar(p,
                        location = 'bottom',
                        pad=0.08,
                        shrink=0.7,
                        extend='both',
                        format=formatter,
                        ticks=[0.005,0.01,0.05]
                        ).set_label(
                            label=r"Calcification ($\mathrm{\mu}$mol C m$^{-3}$ d$^{-1}$)",
                            size=24)
            else:
                if depth_integrated==True:    
                    cb = plt.colorbar(p,
                        location = 'bottom',
                        pad=0.08,
                        shrink=0.7,
                        extend='max'
                        ).set_label(
                            label="Calcification (mg C m$^{-2}$ d$^{-1}$)",
                            size=24)
                else:
                    cb = plt.colorbar(p,
                        location = 'bottom',
                        pad=0.08,
                        shrink=0.7,
                        extend='max'
                        ).set_label(
                            label=r"Calcification ($\mathrm{\mu}$mol C m$^{-3}$ d$^{-1}$)",
                            size=24)

    def latdepth(self, variable, 
                 cmap=cm.viridis, 
                 ax=None, 
                 fig=None, 
                 log=False, 
                 vmin=None, 
                 vmax=None, 
                 title=None,
                 norm=None,
                 add_contour=True,
                 region=None):
        
        if fig==None:
            fig = plt.figure(figsize=(20, 10))
        if ax==None:
            ax= plt.axes()

        if region=='PAC':
            PAC_boundsE = {'min_lon': 120, 'max_lon': 180, 'min_lat': -77, 'max_lat': 80}
            PAC_boundsW = {'min_lon': -180, 'max_lon': -70, 'min_lat': -77, 'max_lat': 80}
            cocco_plotE = self.ds[variable].sel(lon=slice(PAC_boundsE['min_lon'], PAC_boundsE['max_lon']),
                                               lat=slice(PAC_boundsE['min_lat'], PAC_boundsE['max_lat']))
            cocco_plotW = self.ds[variable].sel(lon=slice(PAC_boundsW['min_lon'], PAC_boundsW['max_lon']),
                                               lat=slice(PAC_boundsW['min_lat'], PAC_boundsW['max_lat']))
            cocco_plot = xr.concat([cocco_plotE, cocco_plotW], dim='lon')
            cocco_plot = cocco_plot.mean(dim=["lon", "time"])
        elif region=='IND':
            IND_bounds = {'min_lon': 20, 'max_lon': 120, 'min_lat': -77, 'max_lat': 80}
            cocco_plot = self.ds[variable].sel(lon=slice(IND_bounds['min_lon'], IND_bounds['max_lon']),
                                               lat=slice(IND_bounds['min_lat'], IND_bounds['max_lat']))
            cocco_plot = cocco_plot.mean(dim=["lon", "time"])
        elif region=='ATL':
            ATL_bounds = {'min_lon': -70, 'max_lon': 20, 'min_lat': -77, 'max_lat': 80}
            cocco_plot = self.ds[variable].sel(lon=slice(ATL_bounds['min_lon'], ATL_bounds['max_lon']),
                                               lat=slice(ATL_bounds['min_lat'], ATL_bounds['max_lat']))
            cocco_plot = cocco_plot.mean(dim=["lon", "time"])
        else:
            cocco_plot = self.ds[variable].mean(dim=["lon", "time"])  

        cocco_plot = cocco_plot.where(cocco_plot != 0, float('nan'))

        p = cocco_plot.plot(
                        x='lat', y='depth',
                        cmap=cmap,
                        robust=True,
                        vmax=vmax,
                        vmin=vmin,
                        add_colorbar=False,
                        ax = ax,
                        norm=norm)


        if add_contour == True:
            cocco_plot.plot.contour(x='lat', y='depth', levels=np.arange(5,vmax+1,5),
                            ax = ax)

        ax.set_xlabel('Latitude')
        ax.set_ylabel('Depth')

        if title==None:
            ax.set_title(variable, fontdict = {"fontsize": 24})
        else:
            ax.set_title(title, fontdict = {"fontsize": 24})
            if log==True:
                formatter = LogFormatter(10,labelOnlyBase=False,minor_thresholds=(3,2))
                cb = plt.colorbar(p,
                        location = 'bottom',
                        pad=0.12,
                        shrink=0.7,
                        extend='both',
                        format=formatter,
                        ticks=[5,10,50]
                        ).set_label(
                            label=r"Calcification ($\mathrm{\mu}$mol C m$^{-3}$ d$^{-1}$)",
                            size=24)
            else:
                cb = plt.colorbar(p,
                        location = 'bottom',
                        pad=0.05,
                        shrink=0.7,
                        extend='max'
                        ).set_label(
                            label=r"Calcification ($\mathrm{\mu}$mol C m$^{-3}$ d$^{-1}$)",
                            size=24)

        ax.invert_yaxis()

    def depth(self, variable,
            ax=None, 
            fig=None, 
            region=None,
            title=None,
            norm=None):
        
        if fig==None:
            fig = plt.figure(figsize=(10, 15))
        if ax==None:
            ax= plt.axes()

        if region=='PAC':
            PAC_boundsE = {'min_lon': 120, 'max_lon': 180, 'min_lat': -77, 'max_lat': 60}
            PAC_boundsW = {'min_lon': -180, 'max_lon': -70, 'min_lat': -77, 'max_lat': 60}
            cocco_plotE = self.ds[variable].sel(lon=slice(PAC_boundsE['min_lon'], PAC_boundsE['max_lon']),
                                               lat=slice(PAC_boundsE['min_lat'], PAC_boundsE['max_lat']))
            cocco_plotW = self.ds[variable].sel(lon=slice(PAC_boundsW['min_lon'], PAC_boundsW['max_lon']),
                                               lat=slice(PAC_boundsW['min_lat'], PAC_boundsW['max_lat']))
            cocco_plot = xr.concat([cocco_plotE, cocco_plotW], dim='lon')
            cocco_plot = cocco_plot.mean(dim=["lat", "lon", "time"])
        elif region=='IND':
            IND_bounds = {'min_lon': 20, 'max_lon': 120, 'min_lat': -67, 'max_lat': 22}
            cocco_plot = self.ds[variable].sel(lon=slice(IND_bounds['min_lon'], IND_bounds['max_lon']),
                                               lat=slice(IND_bounds['min_lat'], IND_bounds['max_lat']))
            cocco_plot = cocco_plot.mean(dim=["lat", "lon", "time"])
        elif region=='ATL':
            ATL_bounds = {'min_lon': -70, 'max_lon': 20, 'min_lat': -73, 'max_lat': 79}
            cocco_plot = self.ds[variable].sel(lon=slice(ATL_bounds['min_lon'], ATL_bounds['max_lon']),
                                               lat=slice(ATL_bounds['min_lat'], ATL_bounds['max_lat']))
            cocco_plot = cocco_plot.mean(dim=["lat", "lon", "time"])
        else:
            cocco_plot = self.ds[variable].mean(dim=["lat", "lon", "time"])  

        p = cocco_plot.plot.line(y='depth',color='blue',marker='o',ylim=[-1, 201],xlim=[0,25])
        
        if title==None:
            ax.set_title(variable, fontdict = {"fontsize": 18})
        else:
            ax.set_title(title, fontdict = {"fontsize": 18})

        ax.invert_yaxis()
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.xaxis.label.set_visible(False)
        ax.set_ylabel('Depth (m)')

p = plot_xy(ds)
plt.rcParams.update({'font.size':20})
#%%
p.latlon("Calcification",  
       title="Annual Depth Integrated Calcium Carbonate Production (0-200m)", 
       start_depth = 0,
       end_depth = 200,
       depth_integrated = True,
       vmin=0,
       vmax=80,
       log=False)

#%%
p.latlon("Calcification",  
       title="Annual Mean Calcium Carbonate Production (0-10m)", 
       start_depth = 0,
       end_depth = 10,
       depth_integrated = False,
       log=False,
       vmin=0,
       vmax=70)

#%%
p.latlon("Calcification", 
       title="Annual Mean Calcium Carbonate Production (10-30m)", 
       start_depth = 10,
       end_depth = 30,
       depth_integrated = False,
       log=False,
       vmin=0,
       vmax=70)

#%%
p.latlon("Calcification", 
       title="Annual Mean Calcium Carbonate Production (30-75m)", 
       start_depth = 30,
       end_depth = 75,
       depth_integrated = False,
       log=False,
       vmin=0,
       vmax=70)

#%% Plot 2d Latitude x Depth
p.latdepth("Calcification", 
           log=True, 
           title="Global Zonal Annual Mean Calcium Carbonate Production",
           norm=matplotlib.colors.LogNorm(),
           add_contour=True,
           vmin=3,
           vmax=70)

#%% Plot 2d Latitude x Depth
p.latdepth("Calcification", 
           log=True, 
           title="Pacific Zonal Annual Mean Calcium Carbonate Production",
           norm=matplotlib.colors.LogNorm(),
           add_contour=True,
           vmin=3,
           vmax=70,
           region='PAC')
#%% Plot 2d Latitude x Depth
p.latdepth("Calcification", 
           log=True, 
           title="Indian Zonal Annual Mean Calcium Carbonate Production",
           norm=matplotlib.colors.LogNorm(),
           add_contour=True,
           vmin=3,
           vmax=70,
           region='IND')
#%% Plot 2d Latitude x Depth
p.latdepth("Calcification", 
           log=True, 
           title="Atlantic Zonal Annual Mean Calcium Carbonate Production",
           norm=matplotlib.colors.LogNorm(),
           add_contour=True,
           vmin=3,
           vmax=70,
           region='ATL')

#%% Plot 1d Depth
p.depth("Calcification", 
           title=r"Global Annual Mean Calcium Carbonate Production ($\mathrm{\mu}$mol C m$^{-3}$ d$^{-1}$)",
           region=None)







#%%
# Define the resolution of the grid (in degrees)
resolution_lat = 1.0  # 1 degree
resolution_lon = 1.0  # 1 degree

# Calculate the number of cells in latitude and longitude
num_cells_lat = int(157 / resolution_lat)
num_cells_lon = int(360 / resolution_lon)

# Initialize the 2D array to store the areas
area = np.zeros((num_cells_lat, num_cells_lon))

# Earth's radius in kilometers
earth_radius = 6371000.0  # Earth's radius in meters

# Calculate the area of each cell
for lat_index in range(num_cells_lat):
    for lon_index in range(num_cells_lon):
        # Calculate the latitude range of the cell
        lat_bottom = -77 + lat_index * resolution_lat
        lat_top = lat_bottom + resolution_lat

        # Calculate the longitude range of the cell
        lon_left = -180 + lon_index * resolution_lon
        lon_right = lon_left + resolution_lon

        # Calculate the area of the cell using spherical trigonometry
        areas = (np.sin(np.radians(lat_top)) - np.sin(np.radians(lat_bottom))) * \
               (np.radians(lon_right) - np.radians(lon_left)) * earth_radius ** 2

        # Store the area in the array
        area[lat_index, lon_index] = areas

volume = area*5
# Assuming 'latitude' and 'longitude' are dimensions in your dataset
ds['volume'] = (('lat', 'lon'), volume)
days_per_month = 365.25/12
molar_mass_C = 12.01  # Molar mass of Carbon in grams/mol

#%% 
# Total PP
# Convert micromoles C to moles C and then to petagrams
#global_integrated_total_PP = (ds['Primary_Production'] * ds['volume'] * days_per_month).sum(dim=['lat', 'lon', 'depth', 'time'])
#global_integrated_total_petagrams_PP = (global_integrated_total_PP * molar_mass_C) / (1e15 * 1e6)
#print(global_integrated_total_petagrams_PP)

# Total CP
# Convert micromoles C to moles CaCO3 and then to petagrams
global_integrated_total_calcif = (ds['Calcification'] * ds['volume'] * days_per_month).sum(dim=['lat', 'lon', 'depth', 'time'])
global_integrated_total_petagrams_calcif = (global_integrated_total_calcif * molar_mass_C) / (1e15 * 1e6)
print(global_integrated_total_petagrams_calcif)

# Top 100m PP
depth_range = slice(0,100)
top_layer_ds = ds.sel(depth=depth_range)
#surface_int_PP = (top_layer_ds['Primary_Production'] * top_layer_ds['volume'] * days_per_month).sum(dim=['lat', 'lon', 'depth','time'])
#surface_int_petagrams_PP = (surface_int_PP * molar_mass_C) / (1e15 * 1e6)
#print(surface_int_petagrams_PP)

# Top 100m CP
surface_int_CP = (top_layer_ds['Calcification'] * top_layer_ds['volume'] * days_per_month).sum(dim=['lat', 'lon', 'depth','time'])
surface_int_petagrams_CP = (surface_int_CP * molar_mass_C) / (1e15 * 1e6)
print(surface_int_petagrams_CP)

# %%