# BLE Environment Detection

------

### Introduction:

Some applications need to be configured differently based on the type of location. Like a covid tracking application, which might define contact differently in different locations. Or a cellphone can alter its notification tones based on whether the user is indoors or outdoors. So it would be helpful if we can identify the location type. The easiest solution would be using bluetooth. Because almost everyone owns a mobile phone or some other device with BLE (bluetooth low energery) support. In this project we attempt to identify the location type based on the BLE beacons data. We have collected the BLE signals in different locations and use machine learning models to classify the location as indoors, outdoors, public transport or super market. In the end the Random forest model can identify the location type with 92% accuracy.



### Goals:

The goals for this project are:

- Developing a ble beacon tracer which scans for discoverable ble devices
- Then build a classifier to identify the environment based on the collected data
- Finally evaluate the classifier of its performance



### Results:

- BLE beacon scanner built on Android
- A Random forest model is built to identify location
- Classifier identifies the location with 92% accuracy



### Approach:

The approach is an iterative model in which we start by collecting data and preparing it for classification. Then build a model and based on the results we decide to further tweak the parameters of the classifier or collect more data.



### Implementation:

>  ### Importing necessary libraries
>
>  ```python
>  # to load data from json file
>  import json
>  
>  # to store & manipulate data
>  import numpy as np 
>  import pandas as pd
>  
>  # to plot data
>  import matplotlib.pyplot as plt 
>  
>  # different classification models
>  from sklearn.neural_network import MLPClassifier
>  from sklearn.neighbors import KNeighborsClassifier
>  from sklearn.svm import SVC
>  from sklearn.gaussian_process import GaussianProcessClassifier
>  from sklearn.gaussian_process.kernels import RBF
>  from sklearn.tree import DecisionTreeClassifier
>  from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
>  from sklearn.naive_bayes import GaussianNB
>  from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
>  
>  # to plot the decision tree
>  from sklearn import tree
>  ```
>
>  ### Loading Data into a Pandas Dataframe
>
>  ```python
>  # load data from json file to data1 of type dictionary
>  data1 = json.load(open('data.json'))
>  
>  # function to join data from different locations of same type
>  def join_dicts(ds): 
>   new_d = dict()
>   for d in ds:
>       for key in d.keys(): new_d[key] = d[key]
>   return new_d
>  
>  # data from each location type is joined together. For e.g., 'rewe' and 'aldi' belong in 'super_market'
>  data2 = {
>   'public_transport': join_dicts([data1['bus']]), 
>   'outdoors': join_dicts([data1['busstop'], data1['outdoors']]), 
>   'indoors': join_dicts([data1['ashokroom'], data1['harsharoom'], data1['himaniroom'], data1['darpanroom'], data1['adarshroom'], data1['kotaroom']]),
>   'super_market': join_dicts([data1['rewe'], data1['aldi']])
>  }
>  
>  locations = list(data2.keys())
>  print('data collected in location types:\n',locations)
>  print('instances in each location type:\n', [location+': '+str(len(data2[location])) for location in locations])
>  ```
>
>  ### Convert raw data into Features
>
>  ```python
>  # function to convert data from string to a list of attribute values
>  def parse(x): 
>   start = x.find('device=')+7; end = start+x[start:].find(",")
>   device = x[start:end]; x=x[end:]
>   start = x.find('mAdvertiseFlags=')+16; end = start+x[start:].find(",")
>   mAdvertiseFlags = int(x[start:end]); x=x[end:]
>   start = x.find('mTxPowerLevel=')+14; end = start+x[start:].find(",")
>   mTxPowerLevel = int(x[start:end]); x=x[end:]
>   start = x.find('rssi=')+5; end = start+x[start:].find(",")
>   rssi = int(x[start:end]); x=x[end:]
>   start = x.find('timestampNanos=')+15; end = start+x[start:].find(",")
>   timestampNanos = int(x[start:end]);x=x[end:]
>   start = x.find('eventType=')+10; end = start+x[start:].find(",")
>   eventType = int(x[start:end]); x=x[end:]
>   start = x.find('primaryPhy=')+11; end = start+x[start:].find(",")
>   primaryPhy = int(x[start:end]); x=x[end:]
>   start = x.find('secondaryPhy=')+13; end = start+x[start:].find(",")
>   secondaryPhy = int(x[start:end]); x=x[end:]
>   start = x.find('advertisingSid=')+15; end = start+x[start:].find(",")
>   advertisingSid = int(x[start:end]); x=x[end:]
>   start = x.find('txPower=')+8; end = start+x[start:].find(",")
>   txPower = int(x[start:end]); x=x[end:]
>   start = x.find('periodicAdvertisingInterval=')+28; end = start+x[start:].find("}")
>   periodicAdvertisingInterval = x[start:end]; x=x[end:]
>   if mTxPowerLevel<-999: mTxPowerLevel = 0
>   return [ timestampNanos, device, mAdvertiseFlags, mTxPowerLevel, rssi, eventType, primaryPhy, secondaryPhy, advertisingSid, txPower, periodicAdvertisingInterval ]
>  
>  # these are the attributes collected from the ble beacons
>  columns1 = ["timestamp", "device_id", "advertise_flag", "transmission_power", "rssi", "event_type", "primary_phy", "secondary_phy", "advertising_sid", "tx_power", "advertising_interval"]
>  
>  # data3 is a dictionary with location names a keys and the data collected in each location as values as a Pandas dataframe type.
>  data3 = { location: pd.DataFrame([parse(instance) for instance in data2[location].values()], columns=columns1) for location in locations}
>  
>  print('sample data indoors:')
>  data3['indoors'].head()
>  ```
>
>  ### Understanding the data
>
>  ```python
>  # print the unique values of each attribute in each location
>  for location in locations:
>   uniques = []
>   for key in columns1[1:]:
>       unique_values = data3[location][key].unique()
>       uniques.append([key, len(unique_values), unique_values[:20]])
>   print('\n\n',location,':')
>   display(pd.DataFrame(uniques, columns=['attribute', '#unique_values', 'some_uniques']))
>  ```
>
>  The attributes primary_phy, secondary_phy, advertising_sid, tx_power, advertising_interval are useless since they have same value in each instance of data. 
>
>  The remaining attributes:
>
>  * timestamp, device_id, event_type: not immediately useful for classification, but can be used to extract more features.
>  * advertise_flag, transmission_power, rssi: can be used for classification so the useless columns are dropped and the data from all locations is merged by adding location name itself as an attribute.
>
>  ```python
>  # the new attributes after dropping some columns and adding location as a column
>  columns2 = ["location", "timestamp", "device_id", "advertise_flag", "transmission_power", "rssi", "event_type"]
>  
>  # data4 is a Dataframe containing values of attributes from columns2 in all locations
>  data4 = pd.DataFrame(
>     [[location]+list(instance) 
>      for location in locations 
>      for instance in data3[location][columns2[1:]].values], 
>     columns=columns2
>  )
>  
>  print('a sample of the new data:')
>  data4.head()
>  ```
>
>  ### Plotting each attribute vs every other attribute
>
>  ```python
>  # assigning colors to each location
>  colors = {locations[0]:"red", locations[1]:"green", locations[2]:"blue", locations[3]:"yellow"}
>  print(colors)
>  
>  # sampling 1% of random data to plot
>  fig, axs = plt.subplots(4, 4, figsize=(20,20))
>  shuffled_data = data4.sample(frac=0.01).reset_index(drop=True)
>  
>  # plotting attribute_i vs attribute_j 
>  for i in range(4):
>     for j in range(4):
>         for location, x, y in shuffled_data[[columns2[0], columns2[i+3], columns2[j+3]]].values:
>             axs[i,j].scatter(x, y, alpha=1, c=colors[location], s=30)
>         axs[i,j].set_xlabel(columns2[i+3])
>         axs[i,j].set_ylabel(columns2[j+3])
>  plt.show()
>  ```
>
>  These featues doesn't provide a ton of discriminability. Let's test some classifiers anyway.
>
>  ### Testing some classifiers
>
>  ```python
>  # shuffle the data and seperate it into training (70%) and testing data (30%)
>  shuffled_data = data4.sample(frac=1).reset_index(drop=True)
>  n = len(shuffled_data)
>  
>  train_X = shuffled_data[columns2[3:]][:int(n*0.7)] # training attributes or training input
>  train_Y = shuffled_data['location'][:int(n*0.7)] # training labels or training output
>  
>  test_X = shuffled_data[columns2[3:]][int(n*0.7):] # testing attributes or testing input
>  test_Y = shuffled_data['location'][int(n*0.7):] # testing labels or testing output
>  
>  # create the classifiers
>  names = ["Nearest Neighbors", "Random Forest", "Decision Tree"]
>  classifiers = [
>     KNeighborsClassifier(3),
>     RandomForestClassifier(max_depth=5, n_estimators=30, max_features=4),
>     DecisionTreeClassifier(max_depth=5)]
>  
>  # iterate over classifiers and print scores
>  for name, clf in zip(names, classifiers):
>     clf.fit(train_X, train_Y)
>     score = clf.score(test_X, test_Y)
>     print(name, ':', score)
>  ```
>
>  Result: (Accuracy)
>
>  - Nearest Neighbors : 49%
>  - Random Forest : 53%
>  - Decision Tree : 53%
>
>  Not a great result; Which calls for some feature engineering. We get new features by aggregating data of 5 second intervals.
>
>  ### Feature Engineering
>
>  ```python
>  # names of new attributes
>  new_cols = ["n_beacons", "n_uniq_devices", "n_uniq_advflags",
>             "avg_bcn_interval", "min_bcn_interval", "max_bcn_interval",
>             "avg_txpwr", "min_txpwr", "max_txpwr", 
>             "avg_rssi", "min_rssi", "max_rssi",
>             "n_event16", "n_event27"]
>  columns3 = new_cols[:]
>  
>  # function to get the interval of a beacons reappearnce
>  def get_bcn_intervals(instances, n):
>     devices, intervals = np.unique(instances[:,1]), []
>     for device in devices:
>         device_filter = instances[:,1]==device
>         timestamps = instances[:,0][device_filter]
>         curr = timestamps[0]
>         for timestamp in timestamps[1:]: intervals.append(round((timestamp-curr)/1000000000, 3))
>     return intervals, len(intervals)
>  
>  # function to create new features from the data
>  def get_features(instances, n):
>     new_instance = [0 for _ in range(14)]
>     # n_beacons
>     new_instance[0] = n
>     # n_uniq_devices
>     try: new_instance[1] = len(np.unique(instances[:,1]))
>     except: pass
>     # n_uniq_advflags
>     try: new_instance[2] = len(np.unique(instances[:,2]))
>     except: pass
>     # avg_bcn_interval
>     try: 
>         bcn_intervals, n_bcns = get_bcn_intervals(instances[:,0:2], n)
>         if n_bcns!=0: new_instance[3] = round(np.sum(bcn_intervals)/n_bcns, 3)
>         else: new_instance[3] = 0
>     except: pass
>     # min_bcn_interval
>     try: new_instance[4] = round(np.min(bcn_intervals), 3)
>     except: pass
>     # max_bcn_interval
>     try: new_instance[5] = round(np.max(bcn_intervals), 3)
>     except: pass
>     # avg_txpwr
>     try: 
>         txpwrs = instances[:,3]
>         new_instance[6] = round(np.sum(txpwrs)/n,3)
>     except: pass
>     # min_txpwr
>     try: new_instance[7] = np.min(txpwrs)
>     except: pass
>     # max_txpwr
>     try: new_instance[8] = np.max(txpwrs)
>     except: pass
>     # avg_rssi
>     try: 
>         rssi = instances[:,4]
>         new_instance[9] = round(sum(rssi)/n,3)
>     except: pass
>     # min_rssi
>     try: new_instance[10] = np.min(rssi)
>     except: pass
>     # max_rssi
>     try: new_instance[11] = np.max(rssi)
>     except: pass
>     # n_event16
>     events, counts = np.unique(instances[:,5], return_counts=True)
>     ct = dict()
>     for e, c in zip(events, counts): ct[e] = c
>     try: new_instance[12] = ct[16]
>     except: pass
>     # n_event27
>     try: new_instance[13] = ct[27]
>     except: pass
>     return new_instance
>  
>  # data5 is a dictionary containing new engineered data from each location
>  data5, interval = dict(), 5000000000
>  for location in locations:
>     data3[location] = data3[location].sort_values(by=['timestamp'])
>     data5[location], i, instances = list(), 0, data3[location].values
>     n = len(instances)
>     while i<n:
>         start, end = i, i+1
>         while end<n and (instances[end][0]-instances[start][0])<interval: end+=1
>         data5[location].append(get_features(instances[start:end],end-start))
>         i = end
>  data5 = {location: pd.DataFrame(data5[location], columns=new_cols) for location in locations}
>  
>  print('total number of instances:',sum([len(data5[location]) for location in locations]))
>  print('instances of each location:\n', [location+': '+str(len(data5[location])) for location in locations])
>  
>  # data6 gathers data from each location into a Dataframe by adding location as an attribute
>  data6 = pd.DataFrame(
>     [ [location]+list(instance) 
>      for location in locations 
>      for instance in data5[location].values ], 
>     columns=['location']+columns3
>  )
>  data6.head()
>  ```
>
>  ### Plotting the new data
>
>  ```python
>  # assigning colors to each location
>  colors = {locations[0]:"red", locations[1]:"green", locations[2]:"blue", locations[3]:"yellow"}
>  print(colors)
>  
>  # plotting attribute_i vs attribute_j 
>  fig, axs = plt.subplots(14, 14, figsize=(50,50))
>  for i in range(14):
>     for j in range(14):
>         for location in locations:
>             for x, y in data5[location][[columns3[i], columns3[j]]].values[:50]:
>                 axs[i,j].scatter(x, y, alpha=1, c=colors[location], s=10)
>         axs[i,j].set_xlabel(columns3[i])
>         axs[i,j].set_ylabel(columns3[j])
>  plt.show()
>  ```
>
>  We can see somewhat clear clusters forming; Suggesting that the data can be easily classified.
>
>  ### Testing classifiers with new data
>
>  ```python
>  # preparing data for classification
>  attributes = new_cols
>  shuffled_data = data6.sample(frac=1).reset_index(drop=True)
>  n_sample = len(shuffled_data)
>  train_X = shuffled_data[attributes][:int(n_sample*0.7)]
>  train_Y = shuffled_data['location'][:int(n_sample*0.7)]
>  test_X = shuffled_data[attributes][int(n_sample*0.7):]
>  test_Y = shuffled_data['location'][int(n_sample*0.7):]
>  
>  # creating different classifiers
>  names = ["Nearest Neighbors", "Random Forest", "Decision Tree"]
>  classifiers = [
>     KNeighborsClassifier(10),
>     RandomForestClassifier(max_depth=10, n_estimators=20, max_features=10),
>     DecisionTreeClassifier(max_depth=10)]
>  
>  # iterate over classifiers and print their scores
>  for name, clf in zip(names, classifiers):
>     clf.fit(train_X, train_Y)
>     score = clf.score(test_X, test_Y)
>     print(name, ':', score)
>  ```
>
>  Results:
>
>  - Nearest Neighbors : 68%
>  - Random Forest : 93%
>  - Decision Tree : 89%
>
>  

### Evaluation:

![](https://github.com/harshapraneeth/ds-project-team4/blob/main/images/BLE_final.png)
![](https://github.com/harshapraneeth/ds-project-team4/blob/main/images/BLE_final%20(1).png)
![](https://github.com/harshapraneeth/ds-project-team4/blob/main/images/BLE_final%20(2).png)
![](https://github.com/harshapraneeth/ds-project-team4/blob/main/images/BLE_final%20(3).png)
![](https://github.com/harshapraneeth/ds-project-team4/blob/main/images/BLE_final%20(4).png)
![](https://github.com/harshapraneeth/ds-project-team4/blob/main/images/BLE_final%20(5).png)
![](https://github.com/harshapraneeth/ds-project-team4/blob/main/images/BLE_final%20(6).png)
![](https://github.com/harshapraneeth/ds-project-team4/blob/main/images/BLE_final%20(7).png)
![](https://github.com/harshapraneeth/ds-project-team4/blob/main/images/BLE_final%20(8).png)
![](https://github.com/harshapraneeth/ds-project-team4/blob/main/images/BLE_final%20(9).png)
![](https://github.com/harshapraneeth/ds-project-team4/blob/main/images/BLE_final%20(10).png)
![](https://github.com/harshapraneeth/ds-project-team4/blob/main/images/BLE_final%20(11).png)
