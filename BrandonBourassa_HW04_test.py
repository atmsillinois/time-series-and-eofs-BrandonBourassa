# Import Dask so we can run it faster

from dask.distributed import Client, progress
client = Client(dashboard_address=':5555')  # set up local cluster on your laptop
client

import pandas as pd
import xarray as xr

ds = xr.open_dataset('https://thredds.rda.ucar.edu/thredds/dodsC/aggregations/g/ds633.1/2/TP',
                     chunks={'time':'500MB'})

variables = ['Total_column_water_vapour_surface_Mixed_intervals_Average',
        'Sea_surface_temperature_surface_Mixed_intervals_Average']
#select the two needed variables every 4th point to get 1 degree resolution
ds = ds[variables].sel(lat=slice(65, -65, 4), lon=slice(120, 300, 4))

# Save to netcdf
ds.to_netcdf('raw_data.nc')

# Open netcdf file
ds = xr.open_dataset('raw_data.nc')

#Extract variables

sst = ds['Sea_surface_temperature_surface_Mixed_intervals_Average']
total_water = ds['Total_column_water_vapour_surface_Mixed_intervals_Average']

import numpy as np

# Reduce the data resolution to 1 degree, and get only Pacific region for both sst and total water
sst = sst.interp(lat=np.arange(65, -65, -1), lon=np.arange(120, 300, 1))
total_water = total_water.interp(lat=np.arange(65, -65, -1), lon=np.arange(120, 300, 1))

# Calculate sst anomaly

# Extract sst variables from dataset and calculate the mean SST 
sst_mean = sst.mean(dim='time')

# Calculate SST anomalies by subtracting mean pacific sst from actual pacific sst
sst_anom = sst - sst_mean

# Calculate the mean total column water vapor

# Extract sst variables from dataset and calculate the mean SST 
total_water_mean = total_water.mean(dim='time')

# To plot mean sst on map extract variables

# Extract lat and lon
lat = sst['lat']
lon = sst['lon']

# Convert sst from K to F
sst_mean_f =  (9/5) * (sst_mean - 273.15) + 32

# Now lets create the mask with the mask dataset

mask_url = "https://thredds.rda.ucar.edu/thredds/dodsC/files/g/ds633.0/e5.oper.invariant/197901/e5.oper.invariant.128_172_lsm.ll025sc.1979010100_1979010100.nc"
mask = xr.open_dataset(mask_url).sel(latitude=slice(65, -65, 1),longitude=slice(120, 300, 1)).compute()

# Reduce the data resolution to 1 degree, so the degree resolution matches sst anom
mask = mask.interp(latitude=np.arange(65, -65, -1), longitude=np.arange(120, 300, 1))

# Rename coordaintes to lat and lon to match the initial dataset
mask = mask.rename({'latitude':'lat','longitude':'lon'})

# Find where only water is on the map
WaterOnly=total_water_mean.where(mask_squeeze, drop=True)

import matplotlib.pyplot as plt
import cartopy.crs as ccrs


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

# Plot SST on the first subplot
sst_plot = ax1.contourf(lon, lat, sst_mean_f, cmap='coolwarm', transform=ccrs.PlateCarree())
ax1.coastlines()
ax1.gridlines(draw_labels=True)
ax1.set_title('Mean Sea Surface Temperature (°F) 1979-2023')
ax1.set_extent([120, 300, -65, 65], crs=ccrs.PlateCarree())
# Adding colorbar for SST plot
fig.colorbar(sst_plot, ax=ax1, orientation='horizontal', pad=0.05, aspect=50, shrink=0.5, label='SST (°F)')

# Plot TCWV on the second subplot
tcwv_plot = ax2.contourf(lon, lat, WaterOnly, cmap='viridis', transform=ccrs.PlateCarree())
ax2.coastlines()
ax2.gridlines(draw_labels=True)
ax2.set_title('Mean Total Column Water Vapor (TCWV) 1979-2023')
ax2.set_extent([120, 300, -65, 65], crs=ccrs.PlateCarree())
# Adding colorbar for TCWV plot
fig.colorbar(tcwv_plot, ax=ax2, orientation='horizontal', pad=0.05, aspect=50, shrink=0.5, label='Total Column Water Vapor (mm)')

plt.show()

from scipy import signal

# First, calculate the monthly mean anomaly for each point by subtracting the long-term monthly mean from each month's value. 
# This removes the seasonal cycle.

# Handling infinite values by replacing them with NaN
sst = sst.where(np.isfinite(sst), np.nan)
# Calculating climatology (monthly means) and anomalies, skipping NaN values
climatology = sst.groupby('time.month').mean('time', skipna=True)
anomalies = sst.groupby('time.month') - climatology


# Detrending the anomalies. This removes the linear trend from the deseasonalized time series.
def detrend(da):
    da_no_nan = da.fillna(0)
    detrended = xr.apply_ufunc(signal.detrend, da_no_nan, kwargs={'axis': 0}, dask='allowed')
    return detrended.where(~da.isnull())
detrended_anomalies = detrend(anomalies)

detrended_anomalies

from eofs.xarray import Eof
from eofs.examples import example_data_path

# First, prepare a latitude array with cosine weighting
coslat = np.cos(np.deg2rad(detrended_anomalies.coords['lat'].values)).clip(0., 1.)
weights = np.sqrt(coslat)[..., np.newaxis]

# Create an EOF solver. Assuming detrended_anomalies is 3D: time, lat, lon
solver = Eof(detrended_anomalies, weights=weights)

# Calculate the first 5 EOFs and their explained variances
eofs = solver.eofs(neofs=5)
variance_fraction = solver.varianceFraction(neigs=5)

# Plotting
plt.figure(figsize=(15, 12))
for i in range(5):
    plt.subplot(3, 2, i+1)
    plt.contourf(eofs.coords['lon'], eofs.coords['lat'], eofs[i], levels=np.linspace(-1, 1, 11), extend='both', cmap='seismic')
    plt.title(f'EOF {i+1}, Explained Variance: {variance_fraction[i].values*100:.2f}%')
    plt.colorbar()
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

plt.tight_layout()
plt.show()

# Calculate the variance fraction for the first 10 EOFs
variance_fraction = solver.varianceFraction(neigs=10).values * 100  # Convert to percentage

# Generate the EOF indices (1-based indexing for plotting)
eof_indices = range(1, 11)

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(eof_indices, variance_fraction, color='blue')
plt.xlabel('EOF Number')
plt.ylabel('Percent of Variance Explained')
plt.title('Variance Explained by the First 10 EOFs')
plt.xticks(eof_indices)  # Ensure x-ticks match the EOF numbers
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Lets start with reconstructing the SST field using the first 5 EOFs
pcs = solver.pcs(npcs=5, pcscaling=1) # Project the SST anomalies onto the first 5 EOFs to get their principal component time series
reconstructed_sst = solver.reconstructedField(5) # Reconstruct the spatial field from these components.

# Now lets calculate the Pearson's correlation coefficient between the reconstructed and observed SSTs using xr.corr()
correlation_map = xr.corr(reconstructed_sst, detrended_anomalies, dim='time')

# Plot the correlation map
plt.figure(figsize=(10, 6))
correlation_map.plot(cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Pearson Correlation between Reconstructed and Observed SST Anomalies')
plt.show()

# Lets rename the variable so its easier moving forward
tcwv = total_water

# Deseasonalize, detrend, and standardize the total column water vapor

# First, calculate the monthly mean anomaly for each point by subtracting the long-term monthly mean from each month's value. 
# This removes the seasonal cycle.

# Handling infinite values by replacing them with NaN
tcwv = tcwv.where(np.isfinite(sst), np.nan)

# Calculating climatology (monthly means) and anomalies, skipping NaN values
climatology2 = tcwv.groupby('time.month').mean('time', skipna=True)
anomalies2 = tcwv.groupby('time.month') - climatology2


# Detrending the anomalies. This removes the linear trend from the deseasonalized time series.
def detrend(da):
    da_no_nan2 = da.fillna(0)
    detrended2 = xr.apply_ufunc(signal.detrend, da_no_nan2, kwargs={'axis': 0}, dask='allowed')
    return detrended2.where(~da.isnull())
detrended_anomalies2 = detrend(anomalies2)

detrended_anomalies2

# Reconstruct for just eof1
reconstructed_eof1 = solver.reconstructedField(1) # Reconstruct the spatial field from these components.

# Compute Pearson's correlation coefficient between SST EOF1 and precipitation anomalies
correlation_map2 = xr.corr(reconstructed_eof1, detrended_anomalies2, dim='time')

# Set up the map projection and figure
proj = ccrs.PlateCarree(central_longitude=180)  # Central longitude at 180
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': proj})

# Set the extent to focus on the Pacific region
ax.set_extent([120, 300, -65, 65], crs=ccrs.PlateCarree())

# Plot the correlation map
pcm = ax.pcolormesh(correlation_map2['lon'], correlation_map2['lat'], correlation_map2,
                    cmap='coolwarm', vmin=-1, vmax=1, transform=ccrs.PlateCarree())

# Add features to the map for better geographical context
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, color='lightgrey')

# Add gridlines
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

# Add a colorbar
cbar = fig.colorbar(pcm, ax=ax, shrink=0.7, orientation='horizontal', pad=0.05)
cbar.set_label('Pearson Correlation Coefficient', fontsize=14)
cbar.ax.tick_params(labelsize=12)

# Set the title
ax.set_title('Pearson Correlation between SST EOF1 and Precipitation Anomalies', fontsize=16, pad=20)

plt.show()
