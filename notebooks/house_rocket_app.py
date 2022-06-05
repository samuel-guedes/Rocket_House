# Libraries
from multiprocessing.sharedctypes import Value
import folium
import geopandas
import numpy as np
import pandas as pd
import streamlit as st
from operator import index
import plotly.express as px
from email.policy import default

from streamlit_folium   import folium_static
from folium.plugins     import MarkerCluster
from matplotlib.markers import MarkerStyle

#Page setting
st.set_page_config( layout='wide')
st.title( "House Rocket Company")
st.markdown("Welcome to House Rocket Data Analysis")


#FUNCTIONS
@st.cache( allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)
    data['date'] = pd.to_datetime( data['date'] )

    return data

@st.cache( allow_output_mutation=True )
def get_geofile( url ):
    geofile = geopandas.read_file( url )

    return geofile


#load data
df = get_data('/home/samuel/Desktop/repos/Rocket_House/notebooks/kc_house_data.csv')

# get geofile
url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
geofile = get_geofile( url )

#add new features
df['price_ft2'] = df['price'] / df['sqft_lot']

#add filters
f_attributes = st.sidebar.multiselect( 'Select features', df.columns)
f_zipcode  = st.sidebar.multiselect( 'Select Zip Code', df['zipcode'].sort_values().unique() )

if (f_zipcode != [] ) & (f_attributes != []):
    data=df.loc[df['zipcode'].isin( f_zipcode ), f_attributes]

elif (f_zipcode != [] ) & (f_attributes == []):
    data=df.loc[df['zipcode'].isin( f_zipcode ), :]

elif (f_zipcode == [] ) & (f_attributes != []):
    data=df.loc[:, f_attributes]

else:
    data = df.copy().reset_index(drop=True)

st.header( 'Raw Data')
st.dataframe( data )

c1, c2 = st.columns((2,3))
#Creating metrics
df1 = data[['id','zipcode']].groupby('zipcode').count().reset_index()
df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
df3 = data[['sqft_living','zipcode']].groupby('zipcode').mean().reset_index()
df4 = data[['price_ft2','zipcode']].groupby('zipcode').mean().reset_index()

#merge
m1 = pd.merge(df1,df2, on='zipcode', how='inner')
m2 = pd.merge(m1, df3, on='zipcode', how='inner')
metrics = pd.merge(m2, df4, on='zipcode', how='inner')
metrics.columns = ['zipcode', 'total houses', 'Avg price','Avg sqft_living', 'Avg price/ft2']

c1.header('Average Values')
c1.dataframe( metrics, height=500 )

# statistic descriptive
num_attributes = data.select_dtypes(include=['int64','float64'])
media = pd.DataFrame(num_attributes.apply( np.mean ))
mediana = pd.DataFrame(num_attributes.apply( np.median ))
std = pd.DataFrame(num_attributes.apply( np.std ))
max_ = pd.DataFrame(num_attributes.apply( np.max ))
min_ = pd.DataFrame(num_attributes.apply( np.min ))

stats = pd.concat([max_, min_, media, mediana, std], axis=1).reset_index()
stats.columns = ['attributes','max','min', 'mean', 'median', 'std']

c2.header('Descriptive Analysis')
c2.dataframe( stats ) 


#plot  density map

st.title('Map by density')

c1, c2 = st.columns(2)
c1.header( 'Portfolio Density' )

df = data.sample(1000)

# Base Map - Folium
density_map = folium.Map( location = [data['lat'].mean(), data['long'].mean()],
                          default_zoom_start=15 )

marker_cluster =  MarkerCluster().add_to( density_map )
for name, row in df.iterrows():
    folium.Marker( [row['lat'], row['long']],
    popup='Sold ${0} on: {1}. Features: {2} sqft, {3} bedrooms,\
           {4} bathrooms, year built {5}'.format( row['price'],
                                                  row['date'],
                                                  row['sqft_living'],
                                                  row['bedrooms'],
                                                  row['bathrooms'],
                                                  row['yr_built'] )).add_to( marker_cluster )

with c1:
    folium_static( density_map )


# Region Price Map

c2.header( 'Price Density' )

df = data[['price', 'zipcode']].groupby( 'zipcode' ).mean().reset_index()
df.columns = ['ZIP', 'PRICE']

geofile = geofile[geofile['ZIP'].isin( df['ZIP'].tolist() )]

region_price_map = folium.Map( location=[data['lat'].mean(), 
                               data['long'].mean() ],
                               default_zoom_start=15 )

region_price_map.choropleth( data = df,
                             geo_data =geofile,
                             columns=['ZIP', 'PRICE'],
                             key_on='feature.properties.ZIP',
                             fill_color= 'YlOrRd',
                             fill_opacity= 0.7,
                             line_opacity=0.2,
                             legend_name='AVG Price' )
                             
with c2:
    folium_static( region_price_map )

#========================================
#          Price Ditribution 
#========================================

data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')

st.sidebar.title('Commercial Options')
st.title ("Commercial Attributes")

#---Average Price per Year

min_year_built = int( data['yr_built'].min() )
max_year_built = int( data['yr_built'].max() )

st.sidebar.subheader( 'Select Year Built Range')
f_year_built = st.sidebar.select_slider('Year Built',
                                        options=data['yr_built'].sort_values().unique().tolist(),
                                        value=[min_year_built,max_year_built])


st.subheader("Average Price per Year Built")
df = data.loc[(data['yr_built'] >= f_year_built[0]) & (data['yr_built'] <= f_year_built[1])]
df = df[['price','yr_built']].groupby('yr_built').mean().reset_index()

fig = px.line( df, x='yr_built', y='price')
st.plotly_chart( fig, use_container_width=True)

#---Average Price per Day
st.subheader("Average Price per Day")
st.sidebar.subheader('Select Date Range')

min_date = data['date'].min()
max_date = data['date'].max()

f_date = st.sidebar.select_slider('Date', options = data['date'].sort_values().unique().tolist(),
                                 value = [min_date,max_date] )

df = data[['price','date']].groupby('date').mean().reset_index()
df = df.loc[ (df['date']>=f_date[0]) & (df['date']<=f_date[1])]

fig = px.line( df, x='date', y='price')
st.plotly_chart( fig, use_container_width=True)


# Distribution
st.header('Price Distribution')
st.sidebar.subheader('Select Price Range')

#filter
min_price = data['price'].min()
max_price = data['price'].max()

f_price = st.sidebar.select_slider('Price',options=data['price'].sort_values().unique().tolist(),
                                 value=[min_price,max_price] )

df = data.loc[ (data['price']>=f_price[0]) & (data['price']<=f_price[1])]

#data plot
fig = px.histogram( df, x='price', nbins=50 )
st.plotly_chart( fig, use_container_width=True )

#========================================
#          Filtering Attributes 
#========================================

st.sidebar.title( 'Attributes Options' )

#filters
f_bedrooms = st.sidebar.selectbox( 'Max number of bedrooms', data['bedrooms'].sort_values().unique() )
f_bathrooms = st.sidebar.selectbox( 'Max number of bathrooms', data['bathrooms'].sort_values().unique() )

c1,c2 = st.columns( 2 )

#Bedrooms
c1.header( 'Houses per bedrooms')
df = data[ data['bedrooms'] <= f_bedrooms ]
fig = px.histogram( df, x='bedrooms', nbins=19)
c1.plotly_chart( fig, use_container_width=True)

#Bathrooms
c2.header( 'Houses per bathrooms')
df = data[ data['bathrooms'] <= f_bathrooms ]
fig = px.histogram( df, x='bathrooms', nbins=19)
c2.plotly_chart( fig, use_container_width=True)


#filters
f_floors = st.sidebar.selectbox( 'Max number of floors', data['floors'].sort_values().unique() )
f_waterview = st.sidebar.checkbox( 'Only houses with water view' )

c1,c2 = st.columns( 2 )

#Floors
c1.header('Houses per floors')
df = data[ data['floors'] <= f_floors ]
fig = px.histogram( df, x='floors', nbins=19)
c1.plotly_chart( fig, use_container_width=True)

#Water view
if f_waterview:
    df = data[ data['waterfront'] == 1 ]
else:
    df = data.copy()

fig = px.histogram( df, x='waterfront', nbins=10 )
c2.plotly_chart( fig, use_container_width=True )
