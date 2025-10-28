# Import libraries
import streamlit as st
import pandas as pd
import pickle
import warnings
import numpy as np
warnings.filterwarnings('ignore')

st.title("Airline Customer Satisfaction")
st.write("Gain insights into passenger experiences and improve satisfaction through data analysis and surveys.")

st.image('airline.jpg', width = 700, caption = 'Understand your cutomers to improve your airline services!')

dt_pickle = open('dt_airline.pickle', 'rb')
clf = pickle.load(dt_pickle) 
dt_pickle.close()

# Use default dataset for automation
default_df = pd.read_csv('airline.csv')
default_df = default_df.dropna().reset_index(drop=True)

with st.sidebar.form("survey_form"):
    st.header('Airline Customer Satisfaction Survey')
    
    # Part 1: Customer Details
    st.subheader('Part 1: Customer Details')
    st.write('Provide information about the customer flying.')

    customer_type = st.selectbox('What type of customer is this?', options=default_df['customer_type'].unique())
    travel_type = st.selectbox('Is the customer traveling for business or personal reasons?', options=default_df['type_of_travel'].unique())
    flight_class = st.selectbox('In which class will the customer be flying?', options=default_df['class'].unique())
    age = st.number_input('How old is the customer?',
                          min_value=int(default_df['age'].min()),
                          max_value=int(default_df['age'].max()),
                          value=int(default_df['age'].median()))

    # Part 2: Flight Details
    st.subheader('Part 2: Flight Details')
    st.write('Provide details about the customer\'s flight details.')

    flight_distance = st.number_input('How far is the customer flying in miles?',
                                      min_value=int(default_df['flight_distance'].min()),
                                      max_value=int(default_df['flight_distance'].max()),
                                      value=int(default_df['flight_distance'].median()))
    dep_delay = st.number_input('How many minutes was the customer\'s departure delayed? (Enter 0 if not delayed)',
                                min_value=int(default_df['departure_delay_in_minutes'].min()),
                                max_value=int(default_df['departure_delay_in_minutes'].max()),
                                value=int(default_df['departure_delay_in_minutes'].median()))
    arr_delay = st.number_input('How many minutes was the customer\'s arrival delayed? (Enter 0 if not delayed)',
                                min_value=int(default_df['arrival_delay_in_minutes'].min()),
                                max_value=int(default_df['arrival_delay_in_minutes'].max()),
                                value=int(default_df['arrival_delay_in_minutes'].median()))

    # Part 3: Customer Experience
    st.subheader('Part 3: Customer Experience')
    st.write('Provide details about the customer\'s flight experience and satisfaction.')

    seat_comfort = st.radio('How comfortable was the seat for the customer?', 
                            [1,2,3,4,5], 
                            horizontal=True)
    dep_convenient = st.radio('Was the departure/arrival time convenient?', 
                              [1,2,3,4,5], 
                              horizontal=True)
    food_drink = st.radio('How would the customer rate the food and drink?', 
                          [1,2,3,4,5], 
                          horizontal=True)
    gate_location = st.radio('How would the customer rate the gate location?', 
                             [1,2,3,4,5], 
                             horizontal=True)
    wifi_service = st.radio('How would the customer rate the in-flight wifi service?', 
                            [1,2,3,4,5], 
                            horizontal=True)
    entertainment = st.radio('How would the customer rate the inflight entertainment?', 
                             [1,2,3,4,5], 
                             horizontal=True)
    online_support = st.radio('How would the customer rate online support?', 
                              [1,2,3,4,5], 
                              horizontal=True)
    booking = st.radio('How easy was online booking for the customer?', 
                       [1,2,3,4,5],
                         horizontal=True)
    onboard_service = st.radio('How would the customer rate the onboard service?', 
                               [1,2,3,4,5], 
                               horizontal=True)
    leg_room = st.radio('How would the customer rate the leg room service?', 
                        [1,2,3,4,5], 
                        horizontal=True)
    baggage = st.radio('How would the customer rate baggage handling?', 
                       [1,2,3,4,5], 
                       horizontal=True)
    checkin = st.radio('How would the customer rate the check-in service?', 
                       [1,2,3,4,5], 
                       horizontal=True)
    cleanliness = st.radio('How would the customer rate cleanliness?', 
                           [1,2,3,4,5], 
                           horizontal=True)
    boarding = st.radio('How would the customer rate online boarding?', 
                        [1,2,3,4,5], 
                        horizontal=True)

    # Submit button at the bottom
    submitted = st.form_submit_button("Predict")


st.expander("What can you do with this app?").markdown("""
- 📝 Fill Out a Survey: Provide a form for users to fill out their airline satisfaction feedback.
- 🌟 Make Datra-Driven Decisions: Use insights to guide improvements in customer experience.
- 🛠 Intereactive Feautures: Explore data with fully interactive charts and summaries.
""")

# Prediction Results Section
st.markdown("<h1 style='text-align: center;'>Prediction of Customer Satisfaction (Decision Tree)</h1>", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown("<h2 style='text-align: center;'>Prediction Result</h2>", unsafe_allow_html=True)
    if submitted:
        encode_df = default_df.copy()
        encode_df = encode_df.drop(columns=['satisfaction'])

        encode_df.loc[len(encode_df)] = [customer_type, age, travel_type, flight_class,
                                      flight_distance, seat_comfort, dep_convenient, 
                                      food_drink, gate_location, wifi_service, entertainment,
                                      online_support, booking, onboard_service, leg_room,
                                      baggage, checkin, cleanliness, boarding, dep_delay, arr_delay]
        
        encode_dummy_df = pd.get_dummies(encode_df)
        user_encoded_df = encode_dummy_df.tail(1)
        new_prediction = clf.predict(user_encoded_df)

        satisfaction = 'Satisfied' if new_prediction[0] == 1 else 'Dissatisfied'
        if satisfaction == "Satisfied":
            color = "green"
        else:
            color = "red"
        st.markdown(f"<h3 style='text-align: center; color: {color};'>Your predicted satisfaction level is <b>{satisfaction}</b></h3>", unsafe_allow_html=True)
        
        # for confidence probability use
        proba = clf.predict_proba(user_encoded_df)
        pred_label = new_prediction[0]
        classes = clf.classes_
        pred_idx = np.where(classes == pred_label)[0][0]      # finds the matching column
        confidence = float(proba[0, pred_idx]) * 100 
        st.write(f"With a confidence of {confidence}%")
                
        st.markdown("<h3 style='text-align: center; color: light gray;'>Customer Demographic Analysis</h3>", unsafe_allow_html=True)

    # Customer Type Comparison
        percent_customer_type = round((default_df['customer_type'] == customer_type).mean() * 100, 2)
        
        with st.expander("**Customer Type Comparison**"):
            st.write(f"Customer Type: Your selection: {customer_type}")
            st.write(f"Percentage of our fliers with this selection: **{percent_customer_type}%**")

            # Type of travel comparison

        percent_travel_type = round((default_df['type_of_travel'] == travel_type).mean() * 100, 2)

        with st.expander("**Travel Type Comparison**"):
            st.write(f"Customer Type: Your selection: {travel_type}")
            st.write(f"Percentage of our fliers with this selection: **{percent_travel_type}%**")

        # Flight Class Comparison
        percent_flight_class = round((default_df['class'] == flight_class).mean() * 100, 2)

        with st.expander("**Flight Class Comparison**"):
            st.write(f"Customer Type: Your selection: {flight_class}")
            st.write(f"Percentage of our fliers with this selection: **{percent_flight_class}%**")

        # Age Group Comparison
        if age < 18:
            age_group = 'Under 18'
        elif 18 <= age < 30:
            age_group = '18-29'
        elif 30 <= age < 45:
            age_group = '30-44'
        elif 45 <= age < 60:
            age_group = '45-59'
        else:
            age_group = '60 and above'

        age_bins = [0, 18, 30, 45, 60, 100]
        age_labels = ['Under 18', '18-29', '30-44', '45-59', '60 and above']
        default_df['age_group'] = pd.cut(default_df['age'], bins=age_bins, labels=age_labels, right=False)
        percent_age_group = round((default_df['age_group'] == age_group).mean() * 100, 2) 
        with st.expander("**Age Group Comparison**"):
            st.write(f"Customer Age Group: Your selection: {age_group}")
            st.write(f"Percentage of our fliers in this age group: **{percent_age_group}%**")
    else:
        st.info("Please fill out the survey form and click **Predict** to see the results.")


    

    