###############################################################################################
#### Initialization
import pandas as pd
import numpy as np
df = pd.read_csv(filename, header=None, names=col_names, na_values={'col_name':['-1']}, \
    parse_dates=[[0, 1, 2]], index_col='Date')
#   if the first 3 columns are 'year','month','day', then the dataframe would have a single col named
#   'year_month_day' of datatype 'datatime64[ns]'
#   Can use df.index = df['year_month_day'] to reassign this col as the index of df

## EDA == Exploratory Data Analysis

###############################################################################################
#### Basic data exploration
df.shape # shape of dataframe
df.head(7) # print the head part of dataset
df.tail(5) # print the tail part of dataset
df.info() # return data type of each column, and number of non-null values
df.count() # count items for each column 
df.describe() # summary stat of numerical data
    # df.mean(), df.median(), df.std(), df.quantile([0.25, 0.75]), df.min(), df.max()
df['one_col_name'].unique() # unique values in a column
df['one_col_name'].value_counts(dropna=False) # return frequency counts of a column
df['one_col_name'].value_counts(dropna=False).head() # note the result of prev line is a pandas Series
df.idxmax(axis=0) # Or use axis='index'
df.idxmin(axis=1) # Or use axis='columns'
    # indexes of max/min vals for each column/row

###############################################################################################
#### Row & column index manipulation
df.columns # names of all the columns, usually class of Index
    # can be assigned with a list of new names.
df.index # can be assigned with list of new indexes. # row indexes, can be class of Index or DatatimeIndex
df.index = df.index.map(str.lower) # use map to transform the index with a function
    # pandas Index objects are immutable. Must reset the whole indexes of df at once
df = df.set_index(['col1', 'col2']) # change to multiple index (index being of class MultiIndex)
df = df.sort_index() # change multiple index to hierarchical index
    # use tuple to slice multiple index
    # use slice(None) to indicate ":" in the tuple
    # more advanced manipulation of multiple indexes = stacking and unstacking
    # please refer to datacamp course "Manipulating DataFrames with pandas"
df.reindex(ordered_index) # order rows by original index with the order in ordered_index
    # ordered_index = somehow ordered list of original df indices
    # if some item in ordered_index is not in orig_df_indices, there would be a row with that index but NA values
df.sort_index()

###############################################################################################
#### Data visualization for inspection
    # use Bar plots for discrete data counts
    # use Histograms for continuous data counts
df['one_col_name'].plot('hist')
import matplotlib.pyplot as plt
plt.show()
df.boxplot(column='one_numerical_col', by='one_categorical_col') # two columns are involved
df.boxplot(column='population', by='continent') # example of above

###############################################################################################
#### Data extraction & assignment (general)
## direct column access by column name
df["country"] # This is 1D labeled array (class: pandas.core.series.Series)
df[["country"]] # This is dataframe (class: pandas.core.frame.DataFrame)
## row/column access by (built-in) numerircal indexes
df[1:2] # single row as a dataframe...
    # Note: row slicing cannot use a single number, which would be regarded as a col name
df.iloc[1] # row as pandas Series
df.iloc[[1, 2, 3]]
df.iloc[[1,2,3], [0, 1]]
df.iloc[:, [0,1]]
## row/column access by labels
df.loc["RU"] # row as Pandas Series
df.loc[["RU", "IN", "CH"]] # row as Pandas dataframe
df.loc[["RU", "IN", "CH"], ["country", "capital"]] 
df.loc[:, ["country", "capital"]]
## filtering
df[df["area"] > 8]
df[np.logical_and(df["area"] > 8, df["area"] < 10)] # or use the next line
df[(df["area"] > 8 & df["area"] < 10)]
df[np.logical_or(df["area"] < 8, df["area"] > 10)] # or use the next line
df[(df["area"] < 8 | df["area"] > 10)]
## extract df values as ndarrays
data_array = df.values # extract the values as ndarray
col_array = df['col_name'].values # extract column values as ndarray
np.concatenate([arr1, arr2], axis=1)
## create new columns
df['new_col'] = df['existing_col'].str[0] # extract 1st char of 'existing_col' and save as 'new_col' in df
    # note that 'str' here is an attribute name
df['str_split'] = df['existing_col'].str.split('_') # split string with '_' and save as 'str_split' col
df['new_col0'] = df['str_split'].str.get(0)
df['new_col1'] = df['str_split'].str.get(1)
df['new_col'] = df['col_name'].str.upper()
df['new_mask_col'] = df['col_name'].str.contains('given_substring') # Boolean data
for label, row in df.iterrows():
    df.loc[label, "new_col"] = len(row["country"]) # added a new column "new_col" as function of existing data
df["new_col"] = df["country"].apply(len)
df['new_col'] = 0.0 # assign values with broadcasting
## create new copies of existing dataframes
df2 = df.copy()
sorted_df = df.sort_values('col_name') # sort rows (including index) by values in col 'col_name'
## modify existing entries
df.iloc[::3, -1] = np.nan # assign values with broadcasting
## delete row/column
del df['col_name']
df.drop(['col_name1', 'col_name2'], axis=1)
df.drop([1, 2]) # delete rows by numerical indexes
df.drop(index='row_ind') # delete rows by row index
## manage data types
df['treatment b'] = df['treatment b'].astype(str)
df['sex'] = df['sex'].astype('category')
df['treatment a'] = pd.to_numeric(df['treatment a'], errors='coerce') # force conversion
## manage duplicate rows
df = df.drop_duplicates() # drop duplicate rows
## manage missing data (NA/null/NaN)
df_dropped = df.dropna(how='any') # drop rows with NaN values
df['sex'] = df['sex'].fillna(obj_to_fill) # in 'sex' column, fill NaN with obj_to_fill (e.g. mean value)
checker_df = df.notnull() # boolean for each entry of the dataframe
checker_df_reverse = df.isnull() # boolean for each entry of the dataframe
checker_each_col = df.notnull().all() # aggregated for each column
checker_each_col_reverse = df.isnull().any() # aggregated for each column
checker_col = df.one_col_name.notnull() # boolean for the col "one_col_name"

###############################################################################################
#### tidy data
# tidy data principle: rows contain observations, columns form variables
# pd.melt(): solve the problem of columns (names) containing values, instead of variables
#   ... by turning columns into rows
new_df = pd.melt(frame=df, id_vars=list_names_cols, value_vars=['treatment a', 'treatment b'], \
    var_name='treatment', value_name='result')
#   the columns in list_names_cols remain unchanged
#   the 'treatment a' and 'treatment b' cols become values of a new col called 'treatment'
#   the original table values are collected as values of a new col called 'result'
# pivot: opposite of melting
#   ... by taking unique values from a column and create new columns
weather_tidy = weather.pivot(index='date', columns='element', values='value')
#   the levels in 'element' column become new col names
#   if the values are not specified or multiple, the new columns would become hierarchical index
# if there is duplicate conflict, use aggregate function
weather_tidy = weather.pivot(index='date', columns='element', values='value', aggfunc=np.mean)
# more advanced manipulation of multiple indexes = stacking and unstacking
#   please refer to datacamp course "Manipulating DataFrames with pandas"

###############################################################################################
#### Data (table) joining/concatenation (like in SQL)
## concatenate dataframes
vertical_stacked = df1.append(df2) # indices are also stacked
vertical_stacked.reset_index(drop=True) # result would be the same as the following line
vertical_stacked = pd.concat([df1, df2], axis=0, ignore_index=True) # new indexes range from 0 to n_tot
hori_cat = pd.concat([df1, df2], axis=1, join='outer') # rows with the same index would be merged to single row. cols are stacked
hori_cat = pd.concat([df1, df2], axis=1, join='inner') # only return rows with index in both df1 and df2
df1.join(df2, how='inner/outer/left/right') # join by index
## concatenate lots of tables
import glob
csv_files = glob.glob('*.csv')
list_data = [pd.read_csv(filename) for filename in csv_files]
pd.concat(list_data)
## merge data (index is usually ignored)
pd.merge(left=df_state_populations, right=df_state_codes, on=None, left_on='state', right_on='name')
pd.merge(df_bronze, df_gold, on=['NOC', 'Country'], suffixes=['_bronze', '_gold']) # suffixes for colliding col names of each df
pd.merge(df_bronze, df_gold, on=['NOC', 'Country'], suffixes=['_bronze', '_gold'], how='inner/outer/left/right')
pd.merge_ordered(df1, df2, on='Date', how='outer', fill_method='ffill')

###############################################################################################
#### grouping data by categories
df.groupby('group_col')[['col1', 'col2']].count()
    # can use other aggregations like mean, std, sum, first, last, min, max
df.groupby(['group_col1', 'group_col2']).mean() # resulting in multi-level index
df.groupby('group_col')[['col1', 'col2']].agg(['max','min']) # resulting in multi-level new col names
def data_range(series):  # can use custom_fun for groupby
    return series.max() - series.min()
df.groupby('group_col')[['col1', 'col2']].agg(data_range) # can use custom_fun
df.groupby('group_col')[['col1', 'col2']].agg({'col1':'max', 'col2':'min'})
    # different func for different col
def zscore(series):
    return (series - series.mean())/series.std()
df.groupby('group_col')['col1'].transform(zscore).head() # transformation is done by groups
def zscore_multi(group_df): # transform multiple columns
    new_df = pd.DataFrame({'col1': zscore(group_df['col1']),
                        'col2': group_df['col2']})
    return new_df
df.groupby('group_col').apply(zscore_multi).head() # transform multiple columns
mask = df['filter_col'].str.contains('given_substring')
df.groupby(['group_col', mask])['col1'].mean() # 2-level grouping. 2nd level is 'filter_col' w/ 2 groups (True, False)

###############################################################################################
#### use regular expressions
import re
pattern = re.compile('\$\d*\.\d{2}')
result = pattern.match('$17.89') # result will be True
# the following shows how to create new col with customized function
from numpy import NaN
def diff_money(df_row, pattern_in): # complex function to be applied to row
    icost = df_row['Initial Cost']
    tef = df_row['Total Est. Fee']
    if bool(pattern_in.match(icost)) and bool(pattern_in.match(tef)):
        icost, tef = icost.replace("$", ""), tef.replace("$", "")
        return float(icost) - float(tef)
    else:
        return NaN
df_subset['diff'] = df_subset.apply(diff_money, axis=1, pattern_in=pattern) # note that axis could be 0 or 1

###############################################################################################
## work with time series
# ISO 8601 format: yyyy-mm-dd hh:mm:ss
df.loc['2015-2-5'] # would select all rows within this day
df.loc['2015-2-5':'2015-10-15'] # select range
evening_2_11 = pd.to_datetime(['2015-2-11 20:00', '2015-2-11 21:00', '2015-2-11 22:00', '2015-2-11 23:00'])
df.reindex(evening_2_11) # reindex with the given time stamps, not resetting indexes
df.reindex(evening_2_11, method='ffill/bfill') # forward or backword fill NA values
## resample data -- downsampling (sparser time stamps)
df.loc[:,'col_name'].resample('2W', label='right', closed='right').sum()
## resample data -- upsampling (denser time stamps)
df.loc[:,'col_name'].resample('2H').ffill() # forward fill
df.loc[:,'col_name'].resample('2H').bfill() # backward fill
df.loc[:,'col_name'].resample('2H').first().interpolate('linear') # linearly interpolate
## datetime attributes illustration
df['Date'].dt.hour
central = df['Date'].dt.tz_localize('US/Central')
central.dt.tz_convert('US/Eastern')

## arithmetic
week1_range = weather.loc['2013-07-01':'2013-07-07', ['Min TemperatureF', 'Max TemperatureF']]
week1_mean = weather.loc['2013-07-01':'2013-07-07', 'Mean TemperatureF']
week1_range.divide(week1_mean, axis='rows') # divide A by B by rows
week1_mean.pct_change() * 100 # percentage changes

###############################################################################################
#### save df to file
df.to_csv('my_data.csv')
df.to_excel('my_data.xlsx')