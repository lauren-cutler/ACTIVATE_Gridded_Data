import xarray as xr
import pandas as pd
import numpy as np
import datetime
import glob
import os

def openData(file):
    #####read n the input data
    ds_root = xr.open_dataset(file,engine='h5netcdf',drop_variables='000_Readme',
                              phony_dims='access')
    
    dsDataProd = xr.open_dataset(file,engine='h5netcdf',phony_dims='access',
                                 group='DataProducts')
    dsDataProd = dsDataProd.swap_dims({'phony_dim_4':'z'}) #rename the dimension
        
    dsNavData = xr.open_dataset(file,engine='h5netcdf',phony_dims='access',
                                group='Nav_Data')
    dsNavData = dsNavData.swap_dims({'phony_dim_0':'time',
                                     'phony_dim_1':'phony_dim_5'}) #rename the dimensions
    
    dsState = xr.open_dataset(file,engine='h5netcdf',phony_dims='access',
                              group='State')
    dsState = dsState.swap_dims({'phony_dim_0':'time',
                                 'phony_dim_1':'z'}) #rename the dimensions
    
        
    ds = xr.merge([ds_root,dsDataProd,dsNavData,dsState])
    
    variables = list(ds.keys())
    for var in variables:
        tempvar= ds[var]
        if "StandardName" in tempvar.attrs:
            if isinstance(tempvar.attrs["StandardName"], list) ==0:
                tempvar.attrs["StandardName"] = "None"
        if "Description2" in tempvar.attrs:
            if isinstance(tempvar.attrs["Description2"], list) ==0:
                tempvar.attrs["Description2"] = "None" 
        if hasattr(tempvar, 'altitude bin range for median Sas '):
            tempvar.attrs['altitude bin range for median Sas '] = tempvar.attrs['altitude bin range for median Sas '].flatten()
        if hasattr(tempvar, 'altitude bin range for median Sas '):
            tempvar.attrs['altitude bin range for median Sas'] = tempvar.attrs.pop('altitude bin range for median Sas ')
        if hasattr(tempvar, 'median Sas '):
            tempvar.attrs['median Sas'] = tempvar.attrs.pop('median Sas ')
        if hasattr(tempvar, 'threshold '):
            tempvar.attrs['threshold'] = tempvar.attrs.pop('threshold ')
            
    del ds_root
    del dsDataProd
    del dsNavData
    del dsState
    return ds
    
def Gridding_mean(df,var,longitudeLimits,latitudeLimits,timeLimits,ds):
    
    grid = np.empty([len(timeLimits),len(latitudeLimits),len(longitudeLimits)])
    grid[:] = np.nan
    
    columns_to_check = ['ilon', 'ilat','it']
    idxs = df[~df.duplicated(subset=columns_to_check)].index
    
    for i in idxs:
        ilon, ilat, it = df.ilon[i], df.ilat[i], df.it[i]
                 
        df1 = df[(df.ilon == ilon) & (df.ilat == ilat) & (df.it == it)]
        grid[it,ilat,ilon] = df1[var].mean(skipna=True)
    
    da = xr.DataArray(data=grid, dims=['time','lat','lon'],
                      coords={'lon':longitudeLimits,'lat':latitudeLimits,
                              'time':timeLimits}, name = var+'_Mean')         
    da.attrs = ds[var].attrs
    
    del grid
    del columns_to_check
    del idxs
    del df
    del df1
    return(da)
    
def Gridding_med(df,var,longitudeLimits,latitudeLimits,timeLimits,ds):
    
    grid = np.empty([len(timeLimits),len(latitudeLimits),len(longitudeLimits)])
    grid[:] = np.nan
    
    columns_to_check = ['ilon', 'ilat','it']
    idxs = df[~df.duplicated(subset=columns_to_check)].index
    for i in idxs:
        ilon, ilat, it = df.ilon[i], df.ilat[i], df.it[i]
                 
        df1 = df[(df.ilon == ilon) & (df.ilat == ilat) & (df.it == it)]
        grid[it,ilat,ilon] = df1[var].median(skipna=True)
    
    da = xr.DataArray(data=grid, dims=['time','lat','lon'],
                coords={'lon':longitudeLimits,'lat':latitudeLimits,
                'time':timeLimits}, name = var+'_Median')
         
    da.attrs = ds[var].attrs
    
    del grid
    del columns_to_check
    del idxs
    del df
    del df1
    return(da)
    
def Gridding_std(df,var,longitudeLimits,latitudeLimits,timeLimits,ds):
    
    grid = np.empty([len(timeLimits),len(latitudeLimits),len(longitudeLimits)])
    grid[:] = np.nan
    
    columns_to_check = ['ilon', 'ilat','it']
    idxs = df[~df.duplicated(subset=columns_to_check)].index
    for i in idxs:
        ilon, ilat, it = df.ilon[i], df.ilat[i], df.it[i]
                 
        df1 = df[(df.ilon == ilon) & (df.ilat == ilat) & (df.it == it)]
        grid[it,ilat,ilon] = df1[var].std(skipna=True)
    
    da = xr.DataArray(data=grid, dims=['time','lat','lon'],
                coords={'lon':longitudeLimits,'lat':latitudeLimits,
                'time':timeLimits}, name = var+'_StandardDeviation')
         
    da.attrs = ds[var].attrs
    
    del grid
    del columns_to_check
    del idxs
    del df
    del df1
    return(da)

def Gridding_q25(df,var,longitudeLimits,latitudeLimits,timeLimits,ds):
    
    grid = np.empty([len(timeLimits),len(latitudeLimits),len(longitudeLimits)])
    grid[:] = np.nan
    
    columns_to_check = ['ilon', 'ilat','it']
    idxs = df[~df.duplicated(subset=columns_to_check)].index
    for i in idxs:
        ilon, ilat, it = df.ilon[i], df.ilat[i], df.it[i]
                 
        df1 = df[(df.ilon == ilon) & (df.ilat == ilat) & (df.it == it)]
        grid[it,ilat,ilon] = df1[var].quantile(q=0.25)
    
    da = xr.DataArray(data=grid, dims=['time','lat','lon'],
                coords={'lon':longitudeLimits,'lat':latitudeLimits,
                'time':timeLimits}, name = var+'_25%Quantile')
         
    da.attrs = ds[var].attrs
    
    del grid
    del columns_to_check
    del idxs
    del df
    del df1
    return(da)
    
def Gridding_q75(df,var,longitudeLimits,latitudeLimits,timeLimits,ds):
    
    grid = np.empty([len(timeLimits),len(latitudeLimits),len(longitudeLimits)])
    grid[:] = np.nan
    
    columns_to_check = ['ilon', 'ilat','it']
    idxs = df[~df.duplicated(subset=columns_to_check)].index
    for i in idxs:
        ilon, ilat, it = df.ilon[i], df.ilat[i], df.it[i]
                 
        df1 = df[(df.ilon == ilon) & (df.ilat == ilat) & (df.it == it)]
        grid[it,ilat,ilon] = df1[var].quantile(q=0.75)

    da = xr.DataArray(data=grid, dims=['time','lat','lon'],
                coords={'lon':longitudeLimits,'lat':latitudeLimits,
                'time':timeLimits}, name = var+'_75%Quantile')
         
    da.attrs = ds[var].attrs
    
    del grid
    del columns_to_check
    del idxs
    del df
    del df1
    return(da)
    
def Gridding_count(df,var,longitudeLimits,latitudeLimits,timeLimits,ds):
    
    grid = np.empty([len(timeLimits),len(latitudeLimits),len(longitudeLimits)])
    grid[:] = np.nan
    
    columns_to_check = ['ilon', 'ilat','it']
    idxs = df[~df.duplicated(subset=columns_to_check)].index
    for i in idxs:
        ilon, ilat, it = df.ilon[i], df.ilat[i], df.it[i]
                 
        df1 = df[(df.ilon == ilon) & (df.ilat == ilat) & (df.it == it)]
        grid[it,ilat,ilon] = df1[var].count()

    da = xr.DataArray(data=grid, dims=['time','lat','lon'],
                coords={'lon':longitudeLimits,'lat':latitudeLimits,
                'time':timeLimits}, name = var+'_Count')
         
    da.attrs = ds[var].attrs
    
    del grid
    del columns_to_check
    del idxs
    del df
    del df1
    return(da)
    
def Gridding_alt_mean(df,var,longitudeLimits,latitudeLimits,timeLimits,ds,data,alt):
    
    grid = np.empty([len(timeLimits),len(alt),len(latitudeLimits),len(longitudeLimits)])
    grid[:] = np.nan
    
    columns_to_check = ['ilon', 'ilat','it']
    idxs = df[~df.duplicated(subset=columns_to_check)].index
    for i in idxs:
        ilon, ilat, it = df.ilon[i], df.ilat[i], df.it[i]
    
        ids = df[(df.ilon == ilon) & (df.ilat == ilat) & (df.it == it)].index.values
        grid[it,:,ilat,ilon] = np.nanmean(data[ids,:],axis=0)

    da = xr.DataArray(data=grid, dims=['time','alt','lat','lon'],
                  coords={'lon':longitudeLimits,'lat':latitudeLimits,
                          'time':timeLimits,'alt':alt},name = var+'_Mean')
    da.attrs = ds[var].attrs
    
    del grid
    del columns_to_check
    del idxs
    del df
    return(da)
    
def Gridding_alt_med(df,var,longitudeLimits,latitudeLimits,timeLimits,ds,data,alt):
    
    grid = np.empty([len(timeLimits),len(alt),len(latitudeLimits),len(longitudeLimits)])
    grid[:] = np.nan
    
    columns_to_check = ['ilon', 'ilat','it']
    idxs = df[~df.duplicated(subset=columns_to_check)].index
    for i in idxs:
        ilon, ilat, it = df.ilon[i], df.ilat[i], df.it[i]
        
        ids = df[(df.ilon == ilon) & (df.ilat == ilat) & (df.it == it)].index.values
        grid[it,:,ilat,ilon] = np.nanmedian(data[ids,:],axis=0)

    da = xr.DataArray(data=grid, dims=['time','alt','lat','lon'],
                      coords={'lon':longitudeLimits,'lat':latitudeLimits,
                              'time':timeLimits,'alt':alt},name = var+'_Median')
    da.attrs = ds[var].attrs
    
    del grid
    return(da)

def Gridding_alt_std(df,var,longitudeLimits,latitudeLimits,timeLimits,ds,data,alt):
    
    grid = np.empty([len(timeLimits),len(alt),len(latitudeLimits),len(longitudeLimits)])
    grid[:] = np.nan
    
    columns_to_check = ['ilon', 'ilat','it']
    idxs = df[~df.duplicated(subset=columns_to_check)].index
    for i in idxs:
        ilon, ilat, it = df.ilon[i], df.ilat[i], df.it[i]
        
        ids = df[(df.ilon == ilon) & (df.ilat == ilat) & (df.it == it)].index.values
        grid[it,:,ilat,ilon] = np.nanstd(data[ids,:],axis=0)

    da = xr.DataArray(data=grid, dims=['time','alt','lat','lon'],
                      coords={'lon':longitudeLimits,'lat':latitudeLimits,
                              'time':timeLimits,'alt':alt},name = var+'_StandardDeviation')
    da.attrs = ds[var].attrs
    
    del grid
    del columns_to_check
    del idxs
    del df
    return(da)
    
def Gridding_alt_q25(df,var,longitudeLimits,latitudeLimits,timeLimits,ds,data,alt):
    
    grid = np.empty([len(timeLimits),len(alt),len(latitudeLimits),len(longitudeLimits)])
    grid[:] = np.nan
    
    columns_to_check = ['ilon', 'ilat','it']
    idxs = df[~df.duplicated(subset=columns_to_check)].index
    for i in idxs:
        ilon, ilat, it = df.ilon[i], df.ilat[i], df.it[i]
        
        ids = df[(df.ilon == ilon) & (df.ilat == ilat) & (df.it == it)].index.values
        grid[it,:,ilat,ilon] = np.quantile(data[ids,:],q=0.25,axis=0)

    da = xr.DataArray(data=grid, dims=['time','alt','lat','lon'],
                      coords={'lon':longitudeLimits,'lat':latitudeLimits,
                              'time':timeLimits,'alt':alt},name = var+'_25%Quantile')
    da.attrs = ds[var].attrs
    
    del grid
    del columns_to_check
    del idxs
    del df
    return(da)
    
def Gridding_alt_q75(df,var,longitudeLimits,latitudeLimits,timeLimits,ds,data,alt):
    
    grid = np.empty([len(timeLimits),len(alt),len(latitudeLimits),len(longitudeLimits)])
    grid[:] = np.nan
    
    columns_to_check = ['ilon', 'ilat','it']
    idxs = df[~df.duplicated(subset=columns_to_check)].index
    for i in idxs:
        ilon, ilat, it = df.ilon[i], df.ilat[i], df.it[i]
        
        ids = df[(df.ilon == ilon) & (df.ilat == ilat) & (df.it == it)].index.values
        grid[it,:,ilat,ilon] = np.quantile(data[ids,:],q=0.75,axis=0)

    da = xr.DataArray(data=grid, dims=['time','alt','lat','lon'],
                      coords={'lon':longitudeLimits,'lat':latitudeLimits,
                              'time':timeLimits,'alt':alt},name = var+'_75%Quantile')
    da.attrs = ds[var].attrs
    
    del grid
    del columns_to_check
    del idxs
    del df
    return(da)  
    
def Gridding_alt_count(df,var,longitudeLimits,latitudeLimits,timeLimits,ds,data,alt):
    
    grid = np.empty([len(timeLimits),len(alt),len(latitudeLimits),len(longitudeLimits)])
    grid[:] = np.nan
    
    columns_to_check = ['ilon', 'ilat','it']
    idxs = df[~df.duplicated(subset=columns_to_check)].index
    for i in idxs:
        ilon, ilat, it = df.ilon[i], df.ilat[i], df.it[i]
        
        ids = df[(df.ilon == ilon) & (df.ilat == ilat) & (df.it == it)].index.values
        grid[it,:,ilat,ilon] = np.count_nonzero(data[ids,:],axis=0)

    da = xr.DataArray(data=grid, dims=['time','alt','lat','lon'],
                      coords={'lon':longitudeLimits,'lat':latitudeLimits,
                              'time':timeLimits,'alt':alt},name = var+'_Count')
    da.attrs = ds[var].attrs
    
    del grid
    del columns_to_check
    del idxs
    del df
    return(da)
    
def Gridding_other_mean(df,var,longitudeLimits,latitudeLimits,timeLimits,ds,data):
    
    grid = np.empty([len(timeLimits),len(data[0,:]),len(latitudeLimits),len(longitudeLimits)])
    grid[:] = np.nan
    
    columns_to_check = ['ilon', 'ilat','it']
    idxs = df[~df.duplicated(subset=columns_to_check)].index
    for i in idxs:
        ilon, ilat, it = df.ilon[i], df.ilat[i], df.it[i]
        ids = df[(df.ilon == ilon) & (df.ilat == ilat) & (df.it == it)].index.values
        grid[it,:,ilat,ilon] = np.nanmean(data[ids,:],axis=0)
    
    da = xr.DataArray(data=grid, dims=['time','dim_1','lat','lon'],
                      coords={'lon':longitudeLimits,'lat':latitudeLimits,
                              'time':timeLimits}, name = var+'_Mean')
    da.attrs = ds[var].attrs
    
    del grid
    del columns_to_check
    del idxs
    del df
    return(da)
    
def Gridding_other_med(df,var,longitudeLimits,latitudeLimits,timeLimits,ds,data):
    
    grid = np.empty([len(timeLimits),len(data[0,:]),len(latitudeLimits),len(longitudeLimits)])
    grid[:] = np.nan
    
    columns_to_check = ['ilon', 'ilat','it']
    idxs = df[~df.duplicated(subset=columns_to_check)].index
    for i in idxs:
        ilon, ilat, it = df.ilon[i], df.ilat[i], df.it[i]
        ids = df[(df.ilon == ilon) & (df.ilat == ilat) & (df.it == it)].index.values
        grid[it,:,ilat,ilon] = np.nanmedian(data[ids,:],axis=0)
    
    da = xr.DataArray(data=grid, dims=['time','dim_1','lat','lon'],
                      coords={'lon':longitudeLimits,'lat':latitudeLimits,
                              'time':timeLimits}, name = var+'_Median')
    da.attrs = ds[var].attrs
    
    del grid
    del columns_to_check
    del idxs
    del df
    return(da)
    
def Gridding_other_std(df,var,longitudeLimits,latitudeLimits,timeLimits,ds,data):
    
    grid = np.empty([len(timeLimits),len(data[0,:]),len(latitudeLimits),len(longitudeLimits)])
    grid[:] = np.nan
    
    columns_to_check = ['ilon', 'ilat','it']
    idxs = df[~df.duplicated(subset=columns_to_check)].index
    for i in idxs:
        ilon, ilat, it = df.ilon[i], df.ilat[i], df.it[i]
        ids = df[(df.ilon == ilon) & (df.ilat == ilat) & (df.it == it)].index.values
        grid[it,:,ilat,ilon] = np.nanstd(data[ids,:],axis=0)
    
    da = xr.DataArray(data=grid, dims=['time','dim_1','lat','lon'],
                      coords={'lon':longitudeLimits,'lat':latitudeLimits,
                              'time':timeLimits}, name = var+'_StandardDeviation')
    da.attrs = ds[var].attrs
    
    del grid
    del columns_to_check
    del idxs
    del df
    return(da)
    
def Gridding_other_q25(df,var,longitudeLimits,latitudeLimits,timeLimits,ds,data):
    
    grid = np.empty([len(timeLimits),len(data[0,:]),len(latitudeLimits),len(longitudeLimits)])
    grid[:] = np.nan
    
    columns_to_check = ['ilon', 'ilat','it']
    idxs = df[~df.duplicated(subset=columns_to_check)].index
    for i in idxs:
        ilon, ilat, it = df.ilon[i], df.ilat[i], df.it[i]
        ids = df[(df.ilon == ilon) & (df.ilat == ilat) & (df.it == it)].index.values
        grid[it,:,ilat,ilon] = np.quantile(data[ids,:],q=0.25,axis=0)
    
    da = xr.DataArray(data=grid, dims=['time','dim_1','lat','lon'],
                      coords={'lon':longitudeLimits,'lat':latitudeLimits,
                              'time':timeLimits}, name = var+'_25%Quantile')
    da.attrs = ds[var].attrs
    
    del grid
    del columns_to_check
    del idxs
    del df
    return(da)
    
def Gridding_other_q75(df,var,longitudeLimits,latitudeLimits,timeLimits,ds,data):
    
    grid = np.empty([len(timeLimits),len(data[0,:]),len(latitudeLimits),len(longitudeLimits)])
    grid[:] = np.nan
    
    columns_to_check = ['ilon', 'ilat','it']
    idxs = df[~df.duplicated(subset=columns_to_check)].index
    for i in idxs:
        ilon, ilat, it = df.ilon[i], df.ilat[i], df.it[i]
        ids = df[(df.ilon == ilon) & (df.ilat == ilat) & (df.it == it)].index.values
        grid[it,:,ilat,ilon] = np.quantile(data[ids,:],q=0.75,axis=0)
    
    da = xr.DataArray(data=grid, dims=['time','dim_1','lat','lon'],
                      coords={'lon':longitudeLimits,'lat':latitudeLimits,
                              'time':timeLimits}, name = var+'_75%Quantile')
    da.attrs = ds[var].attrs
    
    del grid
    del columns_to_check
    del idxs
    del df
    return(da)

def Gridding_other_count(df,var,longitudeLimits,latitudeLimits,timeLimits,ds,data):
    
    grid = np.empty([len(timeLimits),len(data[0,:]),len(latitudeLimits),len(longitudeLimits)])
    grid[:] = np.nan
    
    columns_to_check = ['ilon', 'ilat','it']
    idxs = df[~df.duplicated(subset=columns_to_check)].index
    for i in idxs:
        ilon, ilat, it = df.ilon[i], df.ilat[i], df.it[i]
        ids = df[(df.ilon == ilon) & (df.ilat == ilat) & (df.it == it)].index.values
        grid[it,:,ilat,ilon] = np.count_nonzero(data[ids,:],axis=0)
    
    da = xr.DataArray(data=grid, dims=['time','dim_1','lat','lon'],
                      coords={'lon':longitudeLimits,'lat':latitudeLimits,
                              'time':timeLimits}, name = var+'_Count')
    da.attrs = ds[var].attrs
    
    del grid
    del columns_to_check
    del idxs
    del df
    return(da)

def main (year1,gridSpacing,time_res,ver):
    path = 'input data path'
    if year1 == '2022':
        prefixF = "ACTIVATE-HSRL2_KingAir_"
    else:
        prefixF = "ACTIVATE-HSRL2_UC12_"
    
    fileListF=sorted(glob.glob(path+year1+"/"+prefixF+"*.h5"))
    
    for iFile in range(len(fileListF)):
        ds = openData(fileListF[iFile])
        basename = os.path.basename(fileListF[iFile])
        
        if year1 == '2022':
            year = basename[23:27]
            month = basename[27:29]
            day = basename[29:31]
        else:
            year=basename[20:24]
            month=basename[24:26]
            day=basename[26:28]
        
        ###Create Datetime into readiable time
        DateTime=ds['gps_time'][:,0]*3600
        hour = 0
        date_midnight = datetime.datetime(int(year),int(month),int(day),hour)
        delta=pd.to_timedelta(DateTime,'s')
        DateTime=DateTime.rename('DateTime')
        DateTime = date_midnight+delta
        
        ds['DateTime']= xr.DataArray(DateTime.values, dims=['time'])
        ds['DateTime'].attrs['units']='seconds since '+year+'-'+month+'-'+day+' 00:00:00 UTC'
        ds['DateTime'].attrs['long_name']='gps_time converted to Datetime'
        
        ###Change "time" to fixed "DateTime"
        ds = ds.swap_dims({'time':'DateTime'})
        
        lons=ds['gps_lon'].values[:,0]  ###longitude to use
        lats=ds['gps_lat'].values[:,0]  ###latitude to use
        alt=ds['Altitude'].values[0,:]  ###altitude to use 
        time = ds['DateTime'].values[:] ###time to use
        
        ###creating time steps for each file
        timeLimits = pd.date_range(time[0],time[-1]+pd.Timedelta(time_res),freq=time_res)
        
        #Dynamic Grid Creation
        #latitudeLimits = np.arange(25,46,gridSpacing) #ACTIVATE latitudes
        #longitudeLimits = np.arange(-80,-56,gridSpacing) #ACTIVATE longitudes
        latmax = np.ceil(np.max(lats))+gridSpacing ###y2
        latmin = np.floor(np.min(lats)) ###y1
        lonmax = np.ceil(np.max(lons))+gridSpacing ###x2
        lonmin = np.floor(np.min(lons)) ###x1
        
        latitudeLimits = np.arange(latmin,latmax,gridSpacing)
        longitudeLimits = np.arange(lonmin,lonmax,gridSpacing)

        ###Get list of variables in dataset
        variables = list(ds.keys())
        
        ####variables that do not neeed to be gridded
        variables.remove('gps_lon')
        variables.remove('gps_lat')
        variables.remove('lat')
        variables.remove('lon')
        variables.remove('gps_alt')
        variables.remove('gps_time')
        variables.remove('UTCtime2')
        variables.remove('AltBinsize')
        variables.remove('Altitude')
        variables.remove('State_Type')

        for var in variables:
            data = ds[var].values
            print(var+'  '+str(variables.index(var)))
            if data.shape == (len(time),1):
                df = pd.DataFrame({var:data[:,0],'time':time,'lat':lats,'lon':lons})
                
                ####create index associated with each gridspacing
                df['ilat'] = (df.lat - latmin) // gridSpacing
                df['ilon'] = (df.lon - lonmin) // gridSpacing
                df['it'] = (df.time - time[0]) // pd.Timedelta(time_res)
                    
                df['ilat']=df.ilat.astype('int')
                df['ilon']=df.ilon.astype('int')
                df['it']=df.it.astype('int')
                
                ds_mean = Gridding_mean(df,var,longitudeLimits,latitudeLimits,timeLimits,ds)
                ds_median = Gridding_med(df,var,longitudeLimits,latitudeLimits,timeLimits,ds)
                ds_std = Gridding_std(df,var,longitudeLimits,latitudeLimits,timeLimits,ds)
                ds_q25 = Gridding_q25(df,var,longitudeLimits,latitudeLimits,timeLimits,ds)
                ds_q75 = Gridding_q75(df,var,longitudeLimits,latitudeLimits,timeLimits,ds)
                ds_count = Gridding_count(df,var,longitudeLimits,latitudeLimits,timeLimits,ds)
             
            elif data.shape == (len(time),len(alt)):
                df = pd.DataFrame({'time':time,'lat':lats,'lon':lons})
                
                ####create index associated with each gridspacing
                df['ilat'] = (df.lat - latmin) // gridSpacing
                df['ilon'] = (df.lon - lonmin) // gridSpacing
                df['it'] = (df.time - time[0]) // pd.Timedelta(time_res)
                    
                df['ilat']=df.ilat.astype('int')
                df['ilon']=df.ilon.astype('int')
                df['it']=df.it.astype('int')
                
                ds_mean = Gridding_alt_mean(df,var,longitudeLimits,latitudeLimits,timeLimits,ds,data,alt)
                ds_median = Gridding_alt_med(df,var,longitudeLimits,latitudeLimits,timeLimits,ds,data,alt)
                ds_std = Gridding_alt_std(df,var,longitudeLimits,latitudeLimits,timeLimits,ds,data,alt)
                ds_q25 = Gridding_alt_q25(df,var,longitudeLimits,latitudeLimits,timeLimits,ds,data,alt)
                ds_q75 = Gridding_alt_q75(df,var,longitudeLimits,latitudeLimits,timeLimits,ds,data,alt)
                ds_count = Gridding_alt_count(df,var,longitudeLimits,latitudeLimits,timeLimits,ds,data,alt)
       
            else:
                df = pd.DataFrame({'time':time,'lat':lats,'lon':lons})
                
                ####create index associated with each gridspacing
                df['ilat'] = (df.lat - latmin) // gridSpacing
                df['ilon'] = (df.lon - lonmin) // gridSpacing
                df['it'] = (df.time - time[0]) // pd.Timedelta(time_res)
                    
                df['ilat']=df.ilat.astype('int')
                df['ilon']=df.ilon.astype('int')
                df['it']=df.it.astype('int')
                
                ds_mean = Gridding_other_mean(df,var,longitudeLimits,latitudeLimits,timeLimits,ds,data)
                ds_median = Gridding_other_med(df,var,longitudeLimits,latitudeLimits,timeLimits,ds,data)
                ds_std = Gridding_other_std(df,var,longitudeLimits,latitudeLimits,timeLimits,ds,data)
                ds_q25 = Gridding_other_q25(df,var,longitudeLimits,latitudeLimits,timeLimits,ds,data)
                ds_q75 = Gridding_other_q75(df,var,longitudeLimits,latitudeLimits,timeLimits,ds,data)
                ds_count = Gridding_other_count(df,var,longitudeLimits,latitudeLimits,timeLimits,ds,data)
            
            DS= xr.merge([ds_mean,ds_median,ds_std,ds_q25,ds_q75,ds_count])
        
            comp = dict(zlib=True, complevel=5)
            encoding = {var: comp for var in DS.data_vars}
            path1 = 'data output path as a string'
            
            if basename.endswith('_L1.h5'):
                DS.to_netcdf(path1+year1+'/HSRL2_'+var+'_'+year+'-'+month+'-'+day+'_L1_'+str(gridSpacing)+'_'+ver+'.nc',
                             mode ='w',format='NETCDF4',encoding=encoding)
            elif basename.endswith('_L2.h5'):
                DS.to_netcdf(path1+year1+'/HSRL2_'+var+'_'+year+'-'+month+'-'+day+'_L2_'+str(gridSpacing)+'_'+ver+'.nc',
                             mode ='w',format='NETCDF4',encoding=encoding)
            else:
                DS.to_netcdf(path1+year1+'/HSRL2_'+var+'_'+year+'-'+month+'-'+day+'_'+str(gridSpacing)+'_'+ver+'.nc',
                             mode ='w',format='NETCDF4',encoding=encoding)
            
    return ()
main('2020',0.25,'5min','v1') #(year,grid spacing, temporal resolution, data version)
