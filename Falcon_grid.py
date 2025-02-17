import xarray as xr
import pandas as pd
import numpy as np
import datetime
import glob

def openFalconData(file,year1):
    with open(file,encoding='iso-8859-1') as f:
        lines = f.readlines()
    metadataLine1 = lines[0]
    metadataLine2 = lines[6]
    
    naValues = str(lines[11]).split(",")[0]
    
    headers, maxLines=metadataLine1.split(",")
    headers=int(headers)
    maxLines=int(maxLines)
    
    ###get variable info
    if year1 == '2020':
        attrsName = lines[12:136]
    elif year1 == '2021':
        attrsName = lines[12:132]
    else:
        attrsName = lines[12:129]
    
    attrsName.insert(0,lines[8]) #make sure to get start time
    var_name = []
    unit = []
    plat = []
    description = []
    for i in range(0,len(attrsName)):
        temp_list = attrsName[i].split(',')
        
        var_name.append(temp_list[0])
        unit.append(temp_list[1])
        
        if len(temp_list) == 2:
            plat.append('missing')
        else:
            plat.append(temp_list[2])
        
        if len(temp_list) <= 3:
            description.append('None')
        else:
            description.append(temp_list[3])   
    
    #create a dataframe
    df_attrs = pd.DataFrame({'var_name':var_name,
                            'unit':unit,
                            'plat':plat,
                            'descrip':description})
    
    colNames= lines[headers-1]
    colNames=colNames.strip()
    colNamesList=list(colNames.split(", "))
    
    year,month,day,extras=metadataLine2.split(",",3)
    year=year.replace(" ", "")
    month=month.replace(" ", "")
    day=day.replace(" ", "")
    
    df = pd.read_csv(file,skiprows=headers-1,header=0,
                      names=colNamesList,na_values=[naValues],
                      encoding='iso-8859-1')
    
    return (df,df_attrs,year,month,day)

def HSRL2Data (file,time_res,year,month,day):
    #####read in the input data      
    dsNavData = xr.open_dataset(file,engine='h5netcdf',phony_dims='access',
                                group='Nav_Data')
    dsNavData = dsNavData.swap_dims({'phony_dim_0':'time',
                                     'phony_dim_1':'phony_dim_5'}) #rename the dimensions
    
    ###Create Datetime into readiable time
    DateTime=dsNavData['gps_time'][:,0]*3600
    hour = 0
    date_midnight = datetime.datetime(int(year),int(month),int(day),hour)
    delta=pd.to_timedelta(DateTime,'s')
    DateTime=DateTime.rename('DateTime')
    DateTime = date_midnight+delta

    time = DateTime.values[:] ###time to use
    
    #create time limites
    timeLimits = pd.date_range(time[0],time[-1]+pd.Timedelta(time_res),freq=time_res)
    
    return(timeLimits)

def checkTimeLimits(df,timeLimits,num):
    ####check to make sure we have all Falcon Data is considered
    delta = pd.to_timedelta(num,'min')
    
    beg_delta = timeLimits[0] - df['Time'][0]
    end_delta = df['Time'].iloc[-1] - timeLimits[-1]
    
    if beg_delta > delta:
        ###find out by how many factors diff is and round up
        dif_beg = (beg_delta/delta)
        add_int = np.ceil(dif_beg)
        
        ####generate the additional time intervals needed
        ####last time limit will be start of original timelimit
        start = timeLimits[0] - delta*add_int
        beg_dates = pd.date_range(start,timeLimits[0],freq=delta)
        
        ####add time limits to the beginning
        timeLimits = beg_dates[:-1].append(timeLimits)
    if df['Time'].iloc[-1] > timeLimits[-1]:
        if end_delta > delta:
            dif_end = (end_delta/delta)
            add_end = np.ceil(dif_end)
            
            ###generate addtional intervals needed
            ###first time limit will be the end of the original time limit
            end = timeLimits[-1] + delta*add_end
            end_dates = pd.date_range(timeLimits[-1],end,freq=delta)
            
            ####add the time limits to the end
            timeLimits = timeLimits.append(end_dates[1:])
        else:
            end = timeLimits[-1] + delta
            end_dates = pd.date_range(timeLimits[-1],end,freq=delta)
            
            ####add the time limits to the end
            timeLimits = timeLimits.append(end_dates[1:])
       
    return(timeLimits)

def LatLonLim(df,gridSpacing):
    #Dynamic Grid Creation
    #latitudeLimits = np.arange(25,46,gridSpacing) #ACTIVATE latitudes
    #longitudeLimits = np.arange(-80,-56,gridSpacing) #ACTIVATE longitudes
    
    lats = df['Latitude_THORNHILL'].values
    lons = df['Longitude_THORNHILL'].values
    
    latmax = np.ceil(np.nanmax(lats))+gridSpacing ###y2
    latmin = np.floor(np.nanmin(lats)) ###y1
    lonmax = np.ceil(np.nanmax(lons))+gridSpacing ###x2
    lonmin = np.floor(np.nanmin(lons)) ###x1
    
    latitudeLimits = np.arange(latmin,latmax,gridSpacing)
    longitudeLimits = np.arange(lonmin,lonmax,gridSpacing)
    
    return(latitudeLimits,longitudeLimits)

def varGridding(df,gridSpacing,time_res,latitudeLimits,longitudeLimits,timeLimits,df_attrs):
    ####get column names so we know what variables to grid
    ####all dataframes will have the same variables so only need to do this once
    variables = df.columns.values.tolist()
    ###remove variables that it wouldn't make sense to grid
    variables.remove('Time_Start')
    variables.remove('Time_Stop')
    variables.remove('Latitude_THORNHILL')
    variables.remove('Longitude_THORNHILL')
    variables.remove('Date_CORRAL')
    variables.remove('LegIndex_CORRAL')
    variables.remove('Time')
    variables.remove('Flight')
    variables.remove('Leg')
    
    ####remove the variables from the attrs dataframe
    df_attrs = df_attrs.drop(index=[0,1,2,3,21,22])
    df_attrs = df_attrs.reset_index(drop=True)
    
    df['Leg'] = pd.to_numeric(df['Leg'].values) #convert from str to int
    
    mean_values = []
    median_values = []
    std_values = []
    q25_values = []
    q75_values = []
    count_values = []
    for var in variables:
        data = df[var].values
        attrs = df_attrs.iloc[variables.index(var)]
        print(var+'  '+str(variables.index(var)))
            
        df_var = pd.DataFrame({var:data,
                               'time':df['Time'].values,
                               'leg':df['Leg'].values,
                               'lat':df['Latitude_THORNHILL'].values,
                               'lon':df['Longitude_THORNHILL'].values})
        
        df_var = df_var.dropna(subset=['lat','lon'])  ###drop where no lat/lon
        
        lats = df['Latitude_THORNHILL'].values
        lons = df['Longitude_THORNHILL'].values
        time = df['Time'].values
        
        ####create index associated with each gridspacing
        df_var['ilat'] = (df_var.lat - np.floor(np.nanmin(lats))) // gridSpacing
        df_var['ilon'] = (df_var.lon - np.floor(np.nanmin(lons))) // gridSpacing
        df_var['it'] = (df_var.time - time[0]) // pd.Timedelta(time_res)
                
        df_var['ilat']=df_var.ilat.astype('int')
        df_var['ilon']=df_var.ilon.astype('int')
        df_var['it']=df_var.it.astype('int')
        
        da_mean = Gridding_mean(df_var,var,longitudeLimits,latitudeLimits,timeLimits,attrs)
        da_median = Gridding_median(df_var,var,longitudeLimits,latitudeLimits,timeLimits,attrs)
        da_std = Gridding_std(df_var,var,longitudeLimits,latitudeLimits,timeLimits,attrs)
        da_q25 = Gridding_q25(df_var,var,longitudeLimits,latitudeLimits,timeLimits,attrs)
        da_q75 = Gridding_q75(df_var,var,longitudeLimits,latitudeLimits,timeLimits,attrs)
        da_count = Gridding_count(df_var,var,longitudeLimits,latitudeLimits,timeLimits,attrs)
        
        mean_values.append(da_mean)
        median_values.append(da_median)
        std_values.append(da_std)
        q25_values.append(da_q25)
        q75_values.append(da_q75)
        count_values.append(da_count)
        
    DS_mean = xr.Dataset({arr.name: arr for arr in mean_values})
    DS_median = xr.Dataset({arr.name: arr for arr in median_values})
    DS_std = xr.Dataset({arr.name: arr for arr in std_values})
    DS_q25 = xr.Dataset({arr.name: arr for arr in q25_values})
    DS_q75 = xr.Dataset({arr.name: arr for arr in q75_values})
    DS_count = xr.Dataset({arr.name: arr for arr in count_values})
    
    DS = xr.merge([DS_mean,DS_median,DS_std,DS_q25,DS_q75,DS_count])    
       
    return(DS,var)

def Gridding_mean(df,var,longitudeLimits,latitudeLimits,timeLimits,attrs):
    
    grid = np.empty([len(timeLimits),14,len(latitudeLimits),len(longitudeLimits)])
    grid[:] = np.nan
    
    columns_to_check = ['ilon','ilat','it','leg']
    idxs = df[~df.duplicated(subset=columns_to_check)].index
    for i in idxs:
        ilon, ilat, it, ileg = df.ilon[i], df.ilat[i], df.it[i], df.leg[i]
                 
        df1 = df[(df.ilon == ilon) & (df.ilat == ilat) & (df.it == it) & (df.leg == ileg)]
        
        grid[it,ileg,ilat,ilon] = df1[var].mean(skipna=True)
    
    da = xr.DataArray(data=grid, dims=['time','leg','lat','lon'],
                      coords={'lon':longitudeLimits,'lat':latitudeLimits,
                              'time':timeLimits,'leg':np.arange(0,14,1)},
                      name = var+'_Mean',
                      attrs=dict(description=attrs['descrip'],
                                 units=attrs['unit'],
                                 origin=attrs['plat']))    
    del grid     
    return(da)

def Gridding_median(df,var,longitudeLimits,latitudeLimits,timeLimits,attrs):
    
    grid = np.empty([len(timeLimits),14,len(latitudeLimits),len(longitudeLimits)])
    grid[:] = np.nan
    
    columns_to_check = ['ilon','ilat','it','leg']
    idxs = df[~df.duplicated(subset=columns_to_check)].index
    for i in idxs:
        ilon, ilat, it, ileg = df.ilon[i], df.ilat[i], df.it[i], df.leg[i]
                 
        df1 = df[(df.ilon == ilon) & (df.ilat == ilat) & (df.it == it) & (df.leg == ileg)]
        
        grid[it,ileg,ilat,ilon] = df1[var].median(skipna=True)
    
    da = xr.DataArray(data=grid, dims=['time','leg','lat','lon'],
                      coords={'lon':longitudeLimits,'lat':latitudeLimits,
                              'time':timeLimits,'leg':np.arange(0,14,1)},
                      name = var+'_Median',
                      attrs=dict(description=attrs['descrip'],
                                 units=attrs['unit'],
                                 origin=attrs['plat']))      
    del grid
    return(da)

def Gridding_std(df,var,longitudeLimits,latitudeLimits,timeLimits,attrs):
    
    grid = np.empty([len(timeLimits),14,len(latitudeLimits),len(longitudeLimits)])
    grid[:] = np.nan
    
    columns_to_check = ['ilon','ilat','it','leg']
    idxs = df[~df.duplicated(subset=columns_to_check)].index
    for i in idxs:
        ilon, ilat, it, ileg = df.ilon[i], df.ilat[i], df.it[i], df.leg[i]
                 
        df1 = df[(df.ilon == ilon) & (df.ilat == ilat) & (df.it == it) & (df.leg == ileg)]
        
        grid[it,ileg,ilat,ilon] = df1[var].std(skipna=True)
    
    da = xr.DataArray(data=grid, dims=['time','leg','lat','lon'],
                      coords={'lon':longitudeLimits,'lat':latitudeLimits,
                              'time':timeLimits,'leg':np.arange(0,14,1)},
                      name = var+'_StandardDeviation',
                      attrs=dict(description=attrs['descrip'],
                                 units=attrs['unit'],
                                 origin=attrs['plat']))       
    del grid     
    return(da)

def Gridding_q25(df,var,longitudeLimits,latitudeLimits,timeLimits,attrs):
    
    grid = np.empty([len(timeLimits),14,len(latitudeLimits),len(longitudeLimits)])
    grid[:] = np.nan
    
    columns_to_check = ['ilon','ilat','it','leg']
    idxs = df[~df.duplicated(subset=columns_to_check)].index
    for i in idxs:
        ilon, ilat, it, ileg = df.ilon[i], df.ilat[i], df.it[i], df.leg[i]
                 
        df1 = df[(df.ilon == ilon) & (df.ilat == ilat) & (df.it == it) & (df.leg == ileg)]
        
        grid[it,ileg,ilat,ilon] = df1[var].quantile(q=0.25)
    
    da = xr.DataArray(data=grid, dims=['time','leg','lat','lon'],
                      coords={'lon':longitudeLimits,'lat':latitudeLimits,
                              'time':timeLimits,'leg':np.arange(0,14,1)},
                      name = var+'_25%Quantile',
                      attrs=dict(description=attrs['descrip'],
                                 units=attrs['unit'],
                                 origin=attrs['plat']))         
    del grid
    return(da)

def Gridding_q75(df,var,longitudeLimits,latitudeLimits,timeLimits,attrs):
    
    grid = np.empty([len(timeLimits),14,len(latitudeLimits),len(longitudeLimits)])
    grid[:] = np.nan
    
    columns_to_check = ['ilon','ilat','it','leg']
    idxs = df[~df.duplicated(subset=columns_to_check)].index
    for i in idxs:
        ilon, ilat, it, ileg = df.ilon[i], df.ilat[i], df.it[i], df.leg[i]
                 
        df1 = df[(df.ilon == ilon) & (df.ilat == ilat) & (df.it == it) & (df.leg == ileg)]
        
        grid[it,ileg,ilat,ilon] = df1[var].quantile(q=0.75)
    
    da = xr.DataArray(data=grid, dims=['time','leg','lat','lon'],
                      coords={'lon':longitudeLimits,'lat':latitudeLimits,
                              'time':timeLimits,'leg':np.arange(0,14,1)},
                      name = var+'_75%Quantile',
                      attrs=dict(description=attrs['descrip'],
                                 units=attrs['unit'],
                                 origin=attrs['plat']))         
    del grid
    return(da)

def Gridding_count(df,var,longitudeLimits,latitudeLimits,timeLimits,attrs):
    
    grid = np.empty([len(timeLimits),14,len(latitudeLimits),len(longitudeLimits)])
    grid[:] = np.nan
    
    columns_to_check = ['ilon','ilat','it','leg']
    idxs = df[~df.duplicated(subset=columns_to_check)].index
    for i in idxs:
        ilon, ilat, it, ileg = df.ilon[i], df.ilat[i], df.it[i], df.leg[i]
                 
        df1 = df[(df.ilon == ilon) & (df.ilat == ilat) & (df.it == it) & (df.leg == ileg)]
        
        grid[it,ileg,ilat,ilon] = df1[var].count()
    
    da = xr.DataArray(data=grid, dims=['time','leg','lat','lon'],
                      coords={'lon':longitudeLimits,'lat':latitudeLimits,
                              'time':timeLimits,'leg':np.arange(0,14,1)},
                      name = var+'_Count',
                      attrs=dict(description=attrs['descrip'],
                                 units=attrs['unit'],
                                 origin=attrs['plat']))            
    del grid
    return(da)
            

def main (year1,gridSpacing,time_res,ver,num):
    path= 'Falcon data path'
    prefixF = 'ACTIVATE-mrg01-HU25_merge_'
    fileListF=sorted(glob.glob(path+year1+'/'+prefixF+'*.ict'))
    
    for iFile in range(len(fileListF)):
        
        df,attrsName,year,month,day = openFalconData(fileListF[iFile],year1)
    
        ####Convert time into datetime
        hour = 0
        date_midnight = datetime.datetime(int(year),int(month),int(day),hour)
    
        StartTime=df['Time_Start']
        delta_start=pd.to_timedelta(StartTime,'s')
        df['Time_Start'] = date_midnight+delta_start
    
        StopTime=df['Time_Stop']
        delta_stop=pd.to_timedelta(StopTime,'s')
        df['Time_Stop'] = date_midnight+delta_stop
    
        df['Time'] = df['Time_Start'] + pd.to_timedelta(0.5,'s')
        
        ######create separate dataframes for multiflight days
        MultiFlights = ['2020-02-28','2020-03-01','2020-03-08','2020-03-12',
                        '2021-03-05','2021-03-12','2021-03-30','2021-04-02',
                        '2021-05-14','2021-05-19','2021-05-21','2021-05-26',
                        '2021-06-02','2021-06-07','2021-06-08','2021-06-26',
                        '2021-06-30',
                        '2021-12-09','2022-01-11','2022-01-12','2022-01-18',
                        '2022-01-19','2022-01-24','2022-01-26','2022-01-27',
                        '2022-02-03','2022-02-15','2022-02-16','2022-02-19',
                        '2022-02-22','2022-03-03','2022-03-04','2022-03-13',
                        '2022-03-14','2022-03-22','2022-03-26','2022-03-29',
                        '2022-05-05','2022-05-16','2022-05-18','2022-05-21',
                        '2022-06-07','2022-06-08','2022-06-10','2022-06-11',
                        '2022-06-13']
    
        flight_check = year+'-'+month+'-'+day
    
        df['Flight'] = df['LegIndex_CORRAL'].astype(str).str[1:4]
        df['Leg'] = df['LegIndex_CORRAL'].astype(str).str[8:10]
    
        ####will need path to get HSRL-2 data
        path_hsrl = 'HSRL-2 input data path'
        if year1 == '2022':
            prefixF = "ACTIVATE-HSRL2_KingAir_"
            ver1 = 'R3'
        else:
            prefixF = "ACTIVATE-HSRL2_UC12_"
            ver1 = 'R4'
        
        if flight_check in MultiFlights:
            groups = df.groupby('Flight')
            dfs = [group for _, group in groups]
        
            df_F1 = dfs[0].reset_index(drop=True)
            df_F2 = dfs[1].reset_index(drop=True)
            #dfs[2] contains all fill values for LegIndex
        
            f1 = path_hsrl+year1+'/'+prefixF+year+month+day+'_'+ver1+'_L1.h5'
            f2 = path_hsrl+year1+'/'+prefixF+year+month+day+'_'+ver1+'_L2.h5'
        
            HtimeLimits_F1 = HSRL2Data(f1,time_res,year,month,day)
            HtimeLimits_F2 = HSRL2Data(f2,time_res,year,month,day)
        
            ###make sure the time limits from HSRL-2 contain all of the time 
            ###in Falcon Data
            timeLimits_F1 = checkTimeLimits(df_F1,HtimeLimits_F1,num)
            timeLimits_F2 = checkTimeLimits(df_F2,HtimeLimits_F2,num)
        
            ###get latitude and longitude limits
            latitudeLimits_F1, longitudeLimits_F1 = LatLonLim(df_F1,gridSpacing)
            latitudeLimits_F2, longitudeLimits_F2 = LatLonLim(df_F2,gridSpacing)
        
            DS_F1,var1 = varGridding(df_F1,gridSpacing,time_res,latitudeLimits_F1,
                                     longitudeLimits_F1,timeLimits_F1,attrsName)
        
            DS_F2,var2 = varGridding(df_F2,gridSpacing,time_res,latitudeLimits_F2,
                                     longitudeLimits_F2,timeLimits_F2,attrsName)
            del df_F1
            del df_F2
            comp = dict(zlib=True, complevel=5)
            encoding1 = {var: comp for var in DS_F1.data_vars}
            encoding2 = {var: comp for var in DS_F2.data_vars}
            path1 = 'output data path'
        
            DS_F1.to_netcdf(path1+year1+'/Falcon_'+year+'-'+month+'-'+day+'_L1_'+str(gridSpacing)+'_'+ver+'.nc',
                         mode ='w',format='NETCDF4',encoding=encoding1)
        
            DS_F2.to_netcdf(path1+year1+'/Falcon_'+year+'-'+month+'-'+day+'_L2_'+str(gridSpacing)+'_'+ver+'.nc',
                         mode ='w',format='NETCDF4',encoding=encoding2)
        
        else:
            groups = df.groupby('Flight')
            dfs = [group for _, group in groups]
            
            df_F1 = dfs[0].reset_index(drop=True)
            #dfs[1] contains all fill values for LegIndex
        
            f1 = path_hsrl+year1+'/'+prefixF+year+month+day+'_'+ver1+'.h5'
        
            HtimeLimits_F1 = HSRL2Data(f1,time_res,year,month,day)
        
            ###make sure the time limits from HSRL-2 contain all of the time 
            ###in Falcon Data
            timeLimits_F1 = checkTimeLimits(df_F1,HtimeLimits_F1,num)
        
            ###get latitude and longitude limits
            latitudeLimits_F1, longitudeLimits_F1 = LatLonLim(df_F1,gridSpacing)
        
            DS_F1,var1 = varGridding(df_F1,gridSpacing,time_res,latitudeLimits_F1,
                                longitudeLimits_F1,timeLimits_F1,attrsName)
            del df_F1
            comp = dict(zlib=True, complevel=5)
            encoding1 = {var: comp for var in DS_F1.data_vars}
            path1 = 'output data path'
            
            DS_F1.to_netcdf(path1+year1+'/Falcon_'+year+'-'+month+'-'+day+'_'+str(gridSpacing)+'_'+ver+'.nc',
                         mode ='w',format='NETCDF4',encoding=encoding1)  
            
    return()

DS = main('2020',0.25,'5min','v1',5)
