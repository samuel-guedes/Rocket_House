from operator import index
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import folium

from streamlit_folium import folium_static

st.set_page_config( layout='wide')


st.title( "House Rocket Company")
st.markdown("Welcome to House Rocket Data Analysis")


#read data
@st.cache( allow_output_mutation=True)

def get_data(path):
    data = pd.read_csv(path)
    data['date'] = pd.to_datetime( data['date'] )

    return data

#load data
df = get_data('/home/samuel/Desktop/repos/Rocket_House/notebooks/kc_house_data.csv')

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


#plot map
st.title("Region Overview")

c1, c2 = st.columns( (1,1) )
c1.header( 'Portfolio Density')

df_map = data.sample(10)

density_map = folium.Map( location=[data['lat'].mean(),
                                    data['long'].mean()],
                                    default_zoom_start=15)

with c1:
    folium_static( density_map )

is_check = st.checkbox('Display Map')

#filters
price_min = int( df['price'].min() )
price_max = int( df['price'].max() )
price_mean = int( df['price'].mean() )

price_slider = st.slider( 'Price Range',
                            price_min,
                            price_max,
                            price_mean)

if is_check:
    #select rows
    houses = df[df['price'] < price_slider][['id','lat','long','price']]

    fig = px.scatter_mapbox( houses,
                          lat = 'lat',
                          lon = 'long',
                          size = 'price',
                          color_continuous_scale=px.colors.cyclical.IceFire,
                          zoom=10,
                          size_max = 15)

    fig.update_layout( mapbox_style = 'open-street-map' )
    fig.update_layout( height=600, margin={'r':0, 't':0,'l':0, 'b':0})
    st.plotly_chart( fig )
