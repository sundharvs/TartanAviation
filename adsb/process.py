
import argparse
import os
import csv
from glob import glob
from collections import defaultdict
import datetime
import pandas as pd
import numpy as np
import ast

from utils import get_runway_transform, convert_frame

class Data:
    def __init__(self,datapath):
        self.path = datapath + ""
        self.base_path = datapath
        self.filelist = [y for x in os.walk(self.path) for y in glob(os.path.join(x[0], '*.csv'))]
        self.data = defaultdict(lambda: defaultdict())
        self.window = 150
        self.R = get_runway_transform()    
        self.filtered_data = defaultdict(lambda: defaultdict())
        self.filtered_id = 0
        self.out = 1
        self.process_data()
        
    
    def process_data(self):
        ##main loop: Reads each file
        for i in self.filelist:
            print(i)
            with open(i, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    if row["ID"] != "" and  row["Range"] != None and row["Bearing"] != None :
                        ID = row["ID"]
                        Range = row["Range"]
                        Bearing = row["Bearing"]
                        Altitude = row["Altitude"]
                        Tail = row["Tail"]
                        k = Range + Bearing + ID # "unique" ID for deduplication
                        if k not in self.data and int(Altitude)<6000 and float(Range)<5:
                            # self.data[k]["Frame"] = datetime.datetime.strptime(row["Date"]+" "+row["Time"], '%m/%d/%Y %H:%M:%S.%f')
                            date, time = map(ast.literal_eval, (row["Date"], row["Time"]))
                            self.data[k]["Frame"] = datetime.datetime(*map(int, date + time[:2]), int(float(time[2])), int((float(time[2]) % 1) * 1e6))
                            self.data[k]["ID"] = ID
                            self.data[k]["Tail"] = Tail
                            self.data[k]["Range"] = Range
                            self.data[k]["Bearing"] = Bearing
                            self.data[k]["Altitude"] = Altitude
                           
                if not self.data:
                    print("Empty Dict")
                    continue
                df = self.convert_to_local_df()
                df_sorted = self.interp_data(df)
                # Wind information gets appended below
#                 print(df_sorted.head())
                utc_timestamps = pd.DataFrame()
                df_sorted["utc"] = df_sorted.index
#                 utc_timestamps["utc"] = df_sorted.index
#                 print(utc_timestamps.type)

#                 utc_timestamps.parallel_apply(self.get_wind, axis=0)
#                 print(df_sorted)
                    
                self.seg_and_save(df_sorted)                
                self.data = defaultdict(lambda: defaultdict())        
     
    def seg_and_save(self,df):
        ## segregates the data into scenes and saves
        filename = self.base_path + "/processed_data/" + str(self.out) + ".txt" 
        print("Filename = ",filename)
        file = open(filename,'w+')
        csv_writer = csv.DictWriter(file, fieldnames=["Frame","ID","Tail","x","y","z","utc"],delimiter = " ")
        first_time = int(df.iloc[0]["Frame"])
        for index , row in df.iterrows():
            last_time = int(row["Frame"])
            if not ((last_time-first_time) > 1):
                row_write = row.to_dict()
                csv_writer.writerow(row_write)
            else:
                file.close()
                self.out = self.out + 1
                filename = self.base_path + "/processed_data/" + str(self.out) + ".txt"
                print(filename)
                file = open(filename,'w')
                csv_writer = csv.DictWriter(file,fieldnames=["Frame","ID","Tail","x","y","z"],delimiter = " ")
            first_time = last_time
        self.out = self.out + 1    
        file.close()
    
    
    def interp_data(self, data):
        df = data.copy()
        df['datetime'] = pd.to_datetime(df['datetime'], format="%m/%d/%Y,%H:%M:%S")
        df.set_index('datetime', inplace=True)  # Set only datetime as index

        # Separate non-numeric columns
        df_non_numeric = df[['ID', 'Tail']]  # Keep only ID and Tail
        df_numeric = df.select_dtypes(include=[np.number])  # Keep only numeric columns

        # Perform resampling on numeric data, grouped by ID
        df_interpol = df_numeric.groupby(df_non_numeric['ID']).resample('S').mean()

        # Interpolate missing values
        df_interpol[['x', 'y', 'z']] = df_interpol[['x', 'y', 'z']].interpolate(limit=60)

        # Reset index
        df_interpol.reset_index(inplace=True)

        # Resample non-numeric columns (keep only first valid 'Tail' entry per ID)
        df_non_numeric_resampled = df_non_numeric.groupby('ID').resample('S').first()
        df_non_numeric_resampled.reset_index(inplace=True)

        # Merge numeric and non-numeric data
        df_sorted = df_interpol.merge(df_non_numeric_resampled, on=['datetime', 'ID'], how='left')

        # Compute frame times
        df_sorted["time"] = df_sorted["datetime"]
        first = df_sorted["time"].iloc[0]
        df_sorted["Frame"] = (df_sorted["time"] - first).dt.total_seconds().astype(int)
        df_sorted.drop(columns=["time"], inplace=True)

        df_sorted = df_sorted.dropna()
        return df_sorted
    def convert_to_local_df(self):
        ##converts data to local frame
        df = pd.DataFrame.from_dict(self.data,orient='index')
        data = pd.DataFrame()
        data["datetime"] = df["Frame"]
        data['z'] = df.apply(lambda x: float(x["Altitude"])*0.3048/1000.0, axis =1)
        df["pos"] = df.apply(lambda x : convert_frame(float(x["Range"]),float(x["Bearing"]),self.R),axis = 1)
        df[["x","y"]] = pd.DataFrame(df.pos.tolist(),index = df.index)
        data['x'] = df.apply(lambda l: l.x[0],axis =1)
        data['y'] = df.apply(lambda l: l.y[0],axis =1)
        data['ID'] = df["ID"]
        data['Tail'] = df["Tail"]
        
        return data

            
            
if __name__ == '__main__':
    
    ##Dataset params
    parser = argparse.ArgumentParser(description='Train TrajAirNet model')
    parser.add_argument('--dataset_folder',type=str,default='/adsb/data/kbtp/raw')
    parser.add_argument('--dataset_name',type=str,default='/08-01-20')

    args=parser.parse_args()

    data_path = os.getcwd() + args.dataset_folder + args.dataset_name 
    print("Processing data from ",data_path)
    data = Data(data_path)
    data.process_data()
            
            

      
