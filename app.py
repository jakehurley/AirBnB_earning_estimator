#importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px





#page title
st.title('London AirBnB Tool')

st.write('Enter data regarding your property to render an estimate of the price per night \
        you can expect to charge and an estimate of the minimum revenue generated per month')

#London borough locations
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

#types of properties         
properties = ['Entire home/apt', 'Private room', 'Shared room']

#selectbox for London neighbourhoods
neighbourhood = st.sidebar.selectbox('Select your London neighbourhood', locations)

#slider for number of people accommodated
accom = st.sidebar.select_slider('How many can your listing accommodate for?',[1,2,3,4,5,6])

#number of bedrooms slider
beds = st.sidebar.select_slider('How many bedrooms does your listing have?', [0,1,2,3])

#selectbox for property types
property = st.sidebar.selectbox('What type of property is your lisiting?', properties)

#creating dataframe to store inputted values
location_df = pd.DataFrame(np.zeros([1, len(locations)], dtype = int), columns = locations)
property_df = pd.DataFrame(np.zeros([1, len(properties)], dtype = int), columns = properties)
df = pd.concat([property_df, location_df], axis = 1)

#loading scalers/models/data
"""
with open('xgb_rev.pkl', 'rb') as files:
   xgbreg_revenue = pickle.load(files)

with open('ridge_price.pkl', 'rb') as files:
    ridge_price = pickle.load(files)

with open('price_transformer_train.pkl', 'rb') as files:        
    price_transformer_train = pickle.load(files)

with open('price_transformer_target.pkl', 'rb') as files:
    price_transformer_target = pickle.load(files)

with open('revenue_transformer_target.pkl', 'rb') as files:
    revenue_transformer_target = pickle.load(files)  

with open('dataframe.pkl', 'rb') as files:
    data = pickle.load(files)      
"""
    
#button to predict outcome
ok = st.button("Predict price and revenue")

#button execution
if ok: 
    #creating dataframe
    df.insert(0, 'accommodates', accom)
    df.insert(1, 'bedrooms', beds)
    df[property] = 1
    df[neighbourhood] = 1

    #transforming data for price predictor
    df_scaled = price_transformer_train.transform(df)
    
    #predict price with loaded model
    price_pred_scaled = ridge_price.predict(df_scaled)

    #inverse transforming the estimated price 
    price_pred = price_transformer_target.inverse_transform(price_pred_scaled)

    #displaying predicted price
    st.subheader(f'The estimated typical price for a listing of this type is: ${price_pred[0][0]:.2f}')

    #if statement to show message if no data found
    if len(data[(data['accommodates'] == accom)  & (data[property] == 1) & (data[neighbourhood]== 1)]) == 0:
        st.subheader('There are no properties with your selected criteria')

    else:
        #interactive violine plot    
        fig = px.violin(data[(data['accommodates'] == accom)  & (data[property] == 1) & (data[neighbourhood]== 1)],
                        y = 'price', points = 'all', box = True,
                        title = f'Violin plot of price for {property} properties accommodating {accom} in {neighbourhood}' )
        st.plotly_chart(fig)

    #predicting revenue given the estimated price
    df_scaled = np.insert(df_scaled, 2, price_pred_scaled[0][0])    
    rev_pred_scaled = xgbreg_revenue.predict(df_scaled.reshape(1,-1))

    #inverse transforming estimated revenue
    rev_pred = revenue_transformer_target.inverse_transform(rev_pred_scaled.reshape(1,-1))

    st.subheader(f'The estimated minimum revenue for a lisitng of this type is: ${rev_pred[0][0]:.2f}')


