#%%
import numpy as np
import xarray as xr
import math

# Load the .nc file
file_path = '/home/mv23682/Documents/data/Kd_490.nc'
data = xr.open_dataset(file_path)

# Extract kd(490) data, assuming the variable name in the .nc file is 'Kd_490'
kd_490 = data['Kd_490'].values  # shape should be (12, 41, 180, 360)

# We need only the first depth level since all values are the same across depths
kd_490_surface = kd_490[:, 0, :, :]  # shape (12, 180, 360)

# Calculate zeu using the formula zeu = ln(0.01) / kd(490)
zeu = -(np.log(0.01) / kd_490_surface)  # shape (12, 180, 360)

# Create a new xarray Dataset to store the result
zeu_data = xr.Dataset(
    {
        "zeu": (("time", "lat", "lon"), zeu)
    },
    coords={
        "time": data.coords["time"],
        "lat": data.coords["lat"],
        "lon": data.coords["lon"]
    }
)

# Save the result to a new .nc file
zeu_data.to_netcdf('/home/mv23682/Documents/data/zeu.nc')

print("Euphotic zone depth calculated and saved to 'zeu.nc'")