###############################################################################################
#### Data importing with pandas 
### from csv files
import pandas as pd
import numpy as np
df = pd.read_csv("path/to/data_file.csv", index_col = 0) # read in as data frame

###############################################################################################
#### from Excel spreadsheets
data = pd.ExcelFile('data_sheet_file.xlsx')
print(data.sheet_names)
df1 = data.parse('sheetname') # extract sheet with sheetname (str)
df2 = data.parse(0) # extract sheet with index

###############################################################################################
#### from SAS files
from sas7bdat import SAS7BDAT
with SAS7BDAT('urbanpop.sas7bdat') as file:
    df_sas = file.to_data_frame()

###############################################################################################
#### from Stata files
data = pd.read_stata('urbanpop.dta')

###############################################################################################
#### Data importing with h5py
### from HDF5 files
import h5py
filename = 'H-H1_LOSC_4_V1-815411200-4096.hdf5'
data = h5py.File(filename, 'r') # 'r' is to read
# Note: for more details please refer to datacamp course "Importing Data in Python (part 1)"

###############################################################################################
#### Data importing with numpy
### from txt files
import numpy as np
data = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=[0, 2], dtype=str)
# can only specify one datatype

###############################################################################################
#### Data importing with scipy
### from MATLAB files
# scipy.io.loadmat() - read .mat files
# scipy.io.savemat() - write .mat files
import scipy.io
mat = scipy.io.loadmat('workspace.mat')
# mat is a dictionary with keys being variables names and values being objects assigned to variables

###############################################################################################
#### Data importing with pickle
### from pkl files
import pickle
with open('pickled_fruit.pkl', 'rb') as file:
    data = pickle.load(file)

###############################################################################################
#### Data importing with RDMS (relational database maanagement systems)
### from databases
## SQLite database - advantage is fast and simple
## SQLAlchemy - advantage is it works with many RDMS
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Northwind.sqlite')
table_names = engine.table_names()
# below are three ways to conduct query with the engine
con = engine.connect() # build connection to the database with engine
result = con.execute("SELECT * FROM Orders") # execute query
df = pd.DataFrame(result.fetchall()) # fetch the result and save it to a dataframe
df.columns = result.keys() # set the dataframe column names
con.close() # close the connection
# Or, use the context manager
with engine.connect() as con:
    result = con.execute('Query clauses')
    df = pd.DataFrame(result.fetchmany(size=5))
    df.columns = result.keys()
# Or, directly use pandas to query database
df = pd.read_sql_query("Query clauses", engine)

###############################################################################################
#### Data importing with urllib (from webs)
from urllib.request import urlretrieve, urlopen, Request
import os
print(os.getcwd()) # know the cwd
os.chdir('path_of_new_dir') # change to the new wd
url = 'http://archive.ics.uci.edu/ml/machine-learningdatabases/wine-quality/winequality-white.csv'
urlretrieve(url, 'local_file_name.csv')
# URL: "Uniform/Universal Resource Locator"
#   composed of two elements: (1) Protocol identifier - http: (2) Resource name - datacamp.com
# HTTP: "HyperText Transfer Protocol"
# HTTPS: more secured form of HTTP
# Going to a website = sending HTTP request (usually followed by GET request)
# urlretrieve() performs a GET request
# HTML: HyperText Markup Language
# below are two ways to read content from a url
url = "https://www.wikipedia.org/"
request = Request(url)
response = urlopen(request)
html = response.read()
response.close()
# Or, use requests to get web content
import requests
result = requests.get(url)
text = result.text # converted to plain text
# JSON: JavaScript Object Notation (better readability)
json_data = result.json() # converted to json data, whose type is just a dict
for key, value in json_data.items():
    print(key + ":", value)
# use BeautfifulSoup to process the html content
from bs4 import BeautifulSoup
soup = BeautifulSoup(html_doc)
# Note: for more details please refer to datacamp course "Importing Data in Python (part 2)"
# API: Application Programming Interface
# can use website-specific APIs to stream data
#   for more details please refer to datacamp course "Importing Data in Python (part 2)"
