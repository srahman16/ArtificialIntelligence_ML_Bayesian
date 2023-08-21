'''
Title: Script to implement Bayesian GLM for very-high resolution satellite images (single class)
Compiled by: Shahriar Rahman (shahriar.env12@gmail.com)
#Date: 09 August 2023
Version: 2
'''
import numpy as np
import rasterio
import geopandas as gpd
import pymc3 as pm
import theano.tensor as tt
from rasterstats import zonal_stats
import statsmodels.api as sm
import joblib

if __name__ == '__main__':
    gdf = gpd.read_file(r'X:\****\***.shp')
    with rasterio.open(r'X:\****\***.tif') as src:
        # Read the raster data
        raster_data = src.read()

    data = []
    for i, row in gdf.iterrows():
        if row.geometry is None:
            continue
        class_value = 1 if row['CLASS_NAME'] == 'logistics' else 0 # testing with 'bare', 'ocean', 'road'
        raster_values = []
        geometry = row.geometry.__geo_interface__

        for band in raster_data:
            zs = zonal_stats(geometry, band, affine=src.transform, stats=['mean'])
            raster_values.append(zs[0]['mean'])
        data.append([class_value] + raster_values)
    data = np.array(data)

    y = data[:, 0]
    X = data[:, 1:]
    X = sm.add_constant(X) # constant

    with pm.Model() as model:
        beta = pm.Normal('beta', mu=0, sd=1, shape=X.shape[1])
        p = pm.math.sigmoid(tt.dot(X, beta))
        likelihood = pm.Bernoulli('y', p=p, observed=y) #chosen bernoulli distribution
        trace = pm.sample(draws=50, tune=50, chains=2, target_accept=0.8) #test parameters 
    print(pm.summary(trace))

    coefficients = trace['beta'].mean(axis=0)
    band_names = ['constant'] + ['band' + str(i + 1) for i in range(raster_data.shape[0])]
    band_importance = dict(zip(band_names, coefficients))
    sorted_bands = sorted(band_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    print("Band coefficients:")
    for band, coefficient in sorted_bands:
        print(f'{band}: {coefficient}')

    # joblib.dump(X, 'X_raster.joblib')
    # joblib.dump(y, 'y_raster.joblib')
