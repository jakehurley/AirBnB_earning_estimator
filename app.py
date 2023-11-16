import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title('London AirBnB Tool')

locations = ['Barking and Dagenham',
            'Barnet',
            'Bexley',
            'Brent',
            'Bromley',
            'Camden',
            'City of London',
            'Croydon',
            'Ealing',
            'Enfield',
            'Greenwich',
            'Hackney',
            'Hammersmith and Fulham',
            'Haringey',
            'Harrow',
            'Havering',
            'Hillingdon',
            'Hounslow',
            'Islington',
            'Kensington and Chelsea',
            'Kingston upon Thames',
            'Lambeth',
            'Lewisham',
            'Merton',
            'Newham',
            'Redbridge',
            'Richmond upon Thames',
            'Southwark',
            'Sutton',
            'Tower Hamlets',
            'Waltham Forest',
            'Wandsworth',
            'Westminster']
        
properties = ['Entire home/apt', 'Private room', 'Shared room']

neighbourhood = st.selectbox('Select your London neighbourhood', locations)

accom = st.select_slider('How many can your listing accommodate for?',[1,2,3,4,5,6])

beds = st.select_slider('How many bedrooms does your listing have?', [1,2,3])

property = st.selectbox('What type of property is your lisiting?', )

location_df = pd.DataFrame(np.zeros([1, len(locations)], dtype = int), columns = locations)
property_df = pd.DataFrame(np.zeros([1, len(properties)], dtype = int), columns = properties)
df = pd.concat([property_df, location_df], axis = 1)

ok = st.button("Predict price")

if ok: 
    df['accommodates'] = accom
    df['bedrooms'] = beds
    
