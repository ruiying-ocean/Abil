#%%
import numpy as np
import xarray as xr
from yaml import load, Loader

#%% 
# Load datasets
with open('/home/mv23682/Documents/Abil/wiseman2024/ensemble_regressor.yml', 'r') as f:
    model_config = load(f, Loader=Loader)

date = '2024-07-18/'
run_name = '50/'
species = '2024-07-18_cp_ci50'

# Construct file paths using f-strings
ds_path = model_config['local_root'] + model_config['path_out'] + date + run_name + species + ".nc"
zeu_path = '/home/mv23682/Documents/data/zeu.nc'
k_path = '/home/mv23682/Documents/data/Kd_490.nc'

# Load datasets
ds = xr.open_dataset(ds_path)
zeu = xr.open_dataset(zeu_path)
k = xr.open_dataset(k_path)


#%%
# Extract the latitude and longitude coordinates from ds
lat_ds = ds['lat']
lon_ds = ds['lon']

# Subset zeu and k to the latitude and longitude ranges of ds
zeu_subset = zeu.sel(lat=lat_ds, lon=lon_ds)
k_subset = k.sel(lat=lat_ds, lon=lon_ds)

# Extract CR_0 from the surface layer (depth=0) of the ds dataset
CR_0 = ds.sel(depth=0)['Calcification']
k_subset = k_subset.sel(depth=0)['Kd_490']

# Ensure depth values from ds
depths = ds['depth']

# Compute CR_z for each depth
CR_z = xr.concat([CR_0 * np.exp(-k_subset * z) for z in depths], dim='depth')
CR_z = CR_z.assign_coords(depth=depths)
CR_z = CR_z.transpose('time', 'depth', 'lat', 'lon')

#%% Mask depths below z_eu
# Add CR_z back into the original ds dataset
ds['CR_z_HP18'] = CR_z
# Create a mask for non-NaN values in ds['CR_z_HP18']
non_nan_mask = ~np.isnan(ds['CR_z_HP18'])
calcif_non_nan_mask = ~np.isnan(ds['Calcification'])

# Create a mask for the depth condition
depth_condition_mask = ds['CR_z_HP18'].depth > zeu_subset['zeu']
calcif_depth_condition_mask = ds['Calcification'].depth > zeu_subset['zeu']

# Combine the two masks
combined_mask = non_nan_mask & depth_condition_mask
calcif_combined_mask = calcif_non_nan_mask & calcif_depth_condition_mask

# Apply the mask: where the combined condition is true, set the value to 0;
# where false, keep the original value.
masked_CR_z = ds['CR_z_HP18'].where(~combined_mask, other=0)

# Apply the mask: where the combined condition is true, set the value to 0;
# where false, keep the original value.
masked_zeu = ds['Calcification'].where(~calcif_combined_mask, other=0)


# Update ds with masked values
ds['CP_decay'] = masked_CR_z
ds['CP_zeu_mask'] = masked_zeu


# Save the modified dataset to a new NetCDF file
output_path = ds_path.replace('.nc', '_masked.nc')
ds.to_netcdf(output_path)

# %%
