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
       vmax=80)

#%%
p.latlon("Calcification",  
       title="Annual Mean Calcium Carbonate Production (0-10m)", 
       start_depth = 0,
       end_depth = 10,
       vmin=0,
       vmax=70)

#%%
p.latlon("Calcification", 
       title="Annual Mean Calcium Carbonate Production (10-30m)", 
       start_depth = 10,
       end_depth = 30,
       vmin=0,
       vmax=70)

#%%
p.latlon("Calcification", 
       title="Annual Mean Calcium Carbonate Production (30-75m)", 
       start_depth = 30,
       vmin=0,
       vmax=70)

#%% Plot 2d Latitude x Depth
p.latdepth("Calcification", 
           log=True, 
           title="Global Zonal Annual Mean Calcium Carbonate Production",
           norm=matplotlib.colors.LogNorm(),
           vmin=3,
           vmax=70)

#%% Plot 2d Latitude x Depth
p.latdepth("Calcification", 
           log=True, 
           title="Pacific Zonal Annual Mean Calcium Carbonate Production",
           norm=matplotlib.colors.LogNorm(),
           vmin=3,
           vmax=70,
           region='PAC')
#%% Plot 2d Latitude x Depth
p.latdepth("Calcification", 
           log=True, 
           title="Indian Zonal Annual Mean Calcium Carbonate Production",
           norm=matplotlib.colors.LogNorm(),
           vmin=3,
           vmax=70,
           region='IND')
#%% Plot 2d Latitude x Depth
p.latdepth("Calcification", 
           log=True, 
           title="Atlantic Zonal Annual Mean Calcium Carbonate Production",
           norm=matplotlib.colors.LogNorm(),
           vmin=3,
           vmax=70,
           region='ATL')

#%% Plot 1d Depth
p.depth("Calcification", 
           title=r"Global Annual Mean Calcium Carbonate Production ($\mathrm{\mu}$mol C m$^{-3}$ d$^{-1}$)")

#%%

def integrated_total(ds, variable='total', 
                     resolution_lat=1.0, resolution_lon=1.0, depth_w=5, 
                     vol_conversion=1e3, magnitude_conversion=1, molar_mass=1,
                     rate=False):
    
    """
    Estimates global integrated values for a single target. Returns the depth integrated annual total.

    Parameters
    ----------

    PIC data is in pg/l -> currently outputting as monthly average PIC in pg

    ds : xarray.Dataset containing 4-d data of dimensions lat x lon x depth x time
    
    variable : the field to be integrated. Default is 'total' from PIC or POC Abil output

    resolution_lat,resolution_lon : lat/lon resolution in degrees, default is 1 degree

    depth_w : bin depth in meters, default is 5m

    vol_conversion : conversion to m^3, e.g. l to m^3 would be 1e3, default is 1e3 (no conversion)
    
    magnitude_conversion : prefix conversion, e.g. umol to Tmol would be 1e-21, default is 1 (no conversion)

    molar_mass : conversion from mol to grams, default is 1 (no conversion). opt: 12.01 (carbon)

    rate : if input data is in rate of per day, integrates over each month to provide an annual rate (yr^-1)

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> with open('/home/mv23682/Documents/Abil/wiseman2024/ensemble_regressor.yml', 'r') as f:
    ...     model_config = load(f, Loader=Loader)
    >>> run_name = '50'
    >>> species = '2024-05-06_cp_ci50'
    >>> ds = xr.open_dataset(model_config['local_root'] + model_config['path_out'] + run_name + "/" + species + ".nc")
    >>> integrated_total(ds, variable='Calcification', vol_conversion=1, magnitude_conversion=1e-21, molar_mass=12.01, rate=True)


    """

    # Calculate the number of cells in latitude and longitude
    num_cells_lat = int(ds['lat'].size / resolution_lat)   
    num_cells_lon = int(ds['lon'].size / resolution_lon)  
    
    # Retrieve initial latitude and longitude bound
    min_lat = ds['lat'].values[0]
    min_lon = ds['lon'].values[0]

    # Initialize the 2D array to store the areas
    area = np.zeros((num_cells_lat, num_cells_lon))

    earth_radius = 6371000.0  # Earth's radius in meters

    # Calculate the area of each cell
    for lat_index in range(num_cells_lat):
        for lon_index in range(num_cells_lon):
            # Calculate the latitude range of the cell
            lat_bottom = min_lat + lat_index * resolution_lat
            lat_top = lat_bottom + resolution_lat

            # Calculate the longitude range of the cell
            lon_left = min_lon + lon_index * resolution_lon
            lon_right = lon_left + resolution_lon

            # Calculate the area of the grid cell
            areas = earth_radius ** 2 * (np.sin(np.radians(lat_top)) - np.sin(np.radians(lat_bottom))) * \
                    (np.radians(lon_right) - np.radians(lon_left))

            # Store the area in the array
            area[lat_index, lon_index] = areas

    volume = area * depth_w
    ds['volume'] = (('lat', 'lon'), volume)    
    days_per_month = 365.25 / 12  # approx days/month

    if rate:
        total = (ds[variable] * ds['volume'] * days_per_month).sum(dim=['lat', 'lon', 'depth', 'time'])
        total = (total * molar_mass) * vol_conversion * magnitude_conversion
    else:
        total = (ds[variable] * ds['volume']).sum(dim=['lat', 'lon', 'depth', 'time'])
        total = (total * molar_mass) * vol_conversion * magnitude_conversion
    return total

# Example usage:
# ds is your xarray.Dataset containing the variable 'Calcification'
total_CP = integrated_total(ds, variable='Calcification', vol_conversion=1, magnitude_conversion=1e-21, molar_mass=12.01, rate=True)
print(f"Total CP is: {total_CP.values:.2f} PgC/yr")

depth_range = slice(0,100)
top_100m_ds = ds.sel(depth=depth_range)
top_100m_CP = integrated_total(top_100m_ds, variable='Calcification', vol_conversion=1, magnitude_conversion=1e-21, molar_mass=12.01, rate=True)
print(f"Top 100m CP is: {top_100m_CP.values:.2f} PgC/yr")

# %%
