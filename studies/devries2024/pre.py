import pandas as pd
import numpy as np
import glob, os
import xarray as xr
import pickle
import sys

def merge_cascade_env(obs_path="../data/gridded_abundances.csv",
                  env_path="../data/env_data.nc",
                  env_vars=None,
                  out_path="../data/obs_env.csv"):
    """
    Merge observational and environmental datasets based on spatial and temporal indices.

    Parameters
    ----------
    obs_path : str, default="../data/gridded_abundances.csv"
        Path to observational data CSV.
    env_path : str, default="../data/env_data.nc"
        Path to environmental data NetCDF file.
    env_vars : list of str, optional
        List of environmental variables to include in the merge.
    out_path : str, default="../data/obs_env.csv"
        Path to save the merged dataset.

    Returns
    -------
    None
    """
    if env_vars is None:
        env_vars = ["temperature", "sio4", "po4", "no3", "o2", "mld", "DIC",
                    "TA", "irradiance", "chlor_a", "Rrs_547", "Rrs_667", "CI_2",
                    "time", "depth", "lat", "lon"]

    d = pd.read_csv(obs_path)
    d = d.convert_dtypes()

    # Convert to wide format
    d = d.pivot(index=["Latitude", "Longitude", "Depth", "Month", "Year"], 
                    columns="Species", 
                    values="cells L-1").reset_index()

    d = d.groupby(['Latitude', 'Longitude', 'Depth', 'Month']).mean().reset_index()
    d.rename({'Latitude': 'lat', 'Longitude': 'lon', 'Depth': 'depth', 'Month': 'time'}, inplace=True, axis=1)
    d.set_index(['lat', 'lon', 'depth', 'time'], inplace=True)

    print("loading env")
    ds = xr.open_dataset(env_path)
    print("converting to dataframe")
    df = ds.to_dataframe()
    ds = None
    df.reset_index(inplace=True)
    df = df[env_vars]
    df.set_index(['lat', 'lon', 'depth', 'time'], inplace=True)
    print("merging environment")
    out = d.merge(df, how="left", left_index=True, right_index=True)
    out.to_csv(out_path, index=True)
    print("fin")

merge_cascade_env(obs_path = "/home/phyto-2/CASCADE/gridded_datasets/gridded_abundances.csv",
                  env_path= '/home/phyto-2/Abil/studies/devries2024/data/env_data.nc',
                  env_vars = ["temperature", 
                            "sio4", "po4", "no3", 
                            "o2", "mld", "DIC", "TA",
                            "PAR","chlor_a",
                            "time", "depth", 
                            "lat", "lon"],
                    out_path = "/home/phyto-2/Abil/studies/devries2024/data/obs_env.csv")


# d = pd.read_csv("/home/phyto-2/CASCADE/resampled_cellular_datasets/summary_table.csv")
# d.rename(columns={"POC (pg poc) [median]":"pg poc", "PIC (pg pic) [median]":"pg pic", "species":"Target"}, inplace=True)
# d  = d[['Target', 'pg poc', 'pg pic']]
# d.to_csv("/home/phyto-2/Abil/studies/devries2024/data/traits.csv", index=False)

print("fin")