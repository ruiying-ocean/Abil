import pandas as pd
import xarray as xr
import numpy as np

def merge_obs_env(obs_path = "../data/gridded_abundances.csv",
                  env_path = "../data/env_data.nc",
                  env_vars = ["temperature", "si", 
                              "phosphate", "din", 
                              "o2", "mld", "DIC", 
                              "TA", "PAR", 
                              "chlor_a",
                              "time", "depth", 
                              "lat", "lon"],
                  out_path = "../data/obs_env.csv",
                  chunk_size=1e3):  # Chunk size for pandas

    # Read the CSV file in chunks
    chunk_iter = pd.read_csv(obs_path, chunksize=int(chunk_size))
    
    # Open the environmental data once
    print("Loading environmental data")
    ds_env = xr.open_dataset(env_path)
    
    # Select only the relevant environmental variables
    ds_env = ds_env[env_vars]

    # Process each chunk
    for chunk_idx, chunk in enumerate(chunk_iter):
        print(f"Processing chunk {chunk_idx + 1}")

        # Group by necessary columns and calculate the mean for each chunk
        chunk_grouped = chunk.groupby(['Latitude', 'Longitude', 'Depth', 'Month']).mean().reset_index()
        chunk_grouped.rename({'Latitude': 'lat', 'Longitude': 'lon', 'Depth': 'depth', 'Month': 'time'}, inplace=True, axis=1)
        
        # Convert the chunk to an xarray Dataset
        chunk_xr = chunk_grouped.set_index(['lat', 'lon', 'depth', 'time']).to_xarray()

        # Align the observational data with the environmental data based on coordinates
        print(f"Aligning chunk {chunk_idx + 1} with environmental data")
        d_aligned, ds_env_aligned = xr.align(chunk_xr, ds_env, join='left')

        # Merge the aligned datasets
        merged_ds = xr.merge([d_aligned, ds_env_aligned])

        # Convert to DataFrame
        merged_df = merged_ds.to_dataframe().reset_index()

        # Append to CSV file incrementally
        if chunk_idx == 0:
            # Write the header for the first chunk
            merged_df.to_csv(out_path, mode='w', index=False)
        else:
            # Append without writing the header for subsequent chunks
            merged_df.to_csv(out_path, mode='a', header=False, index=False)

    print("Finished")

# Example usage
merge_obs_env(obs_path = "/home/phyto-2/CASCADE/data/output/gridded_abundances.csv",
              env_path= '/media/phyto-2/579ac4be-57ad-4d69-b620-b0427fd30bca/home/phyto/gridded_environmental_data/env_data_nosat.nc',
              env_vars = ["temperature", "si", 
                          "phosphate", "din", 
                          "o2", "mld", "DIC", 
                          "TA", "PAR", 
                          "chlor_a",
                          "time", "depth", 
                          "lat", "lon"],
              out_path = "/media/phyto-2/579ac4be-57ad-4d69-b620-b0427fd30bca/home/phyto/gridded_environmental_data/obs_env.csv")
