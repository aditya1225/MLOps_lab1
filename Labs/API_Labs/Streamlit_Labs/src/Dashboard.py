import json
import requests
import streamlit as st
from pathlib import Path
from streamlit.logger import get_logger

# If you start the fast api server on a different port
# make sure to change the port below
FASTAPI_BACKEND_ENDPOINT = "http://127.0.0.1:8001/"

# Make sure you have california_housing_model.pkl file in FastAPI_Labs/src folder.
# If it's missing run train.py in FastAPI_Labs/src folder
# If your FastAPI_Labs folder name is different, update accordingly in the following path
FASTAPI_HOUSING_MODEL_LOCATION = Path(__file__).resolve().parents[
                                     2] / 'FastAPI_Labs' / 'model' / 'california_housing_model.pkl'

# streamlit logger
LOGGER = get_logger(__name__)


def run():
    # Set the main dashboard page browser tab title and icon
    st.set_page_config(
        page_title="California Housing Price Prediction",
        page_icon="🏠",
    )

    # Build the sidebar first
    # This sidebar context gives access to work on elements in the side panel
    with st.sidebar:
        # Check the status of backend
        try:
            # Make sure fast api is running. Check the lab for guidance on getting
            # the server up and running
            backend_request = requests.get(FASTAPI_BACKEND_ENDPOINT)
            # If backend returns successful connection (status code: 200)
            if backend_request.status_code == 200:
                # This creates a green box with message
                st.success("Backend online ✅")
            else:
                # This creates a yellow bow with message
                st.warning("Problem connecting 😭")
        except requests.ConnectionError as ce:
            LOGGER.error(ce)
            LOGGER.error("Backend offline 😱")
            # Show backend offline message
            st.error("Backend offline 😱")

        st.info("Configure Housing Features")

        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Manual Input", "Upload JSON File"],
            help="Select how you want to provide housing data"
        )

        if input_method == "Manual Input":
            # Set the values using sliders
            median_income = st.slider(
                "Median Income",
                0.5, 15.0, 3.0, 0.1,
                help="Median income in block (in $10,000s)",
                format="%.2f"
            )
            median_house_age = st.slider(
                "House Age",
                1.0, 52.0, 25.0, 1.0,
                help="Median age of houses in block (years)",
                format="%.0f"
            )
            average_rooms = st.slider(
                "Average Rooms",
                1.0, 10.0, 5.0, 0.1,
                help="Average number of rooms per household",
                format="%.2f"
            )
            average_bedrooms = st.slider(
                "Average Bedrooms",
                0.5, 5.0, 1.5, 0.1,
                help="Average number of bedrooms per household",
                format="%.2f"
            )
            population = st.number_input(
                "Population",
                min_value=3,
                max_value=40000,
                value=1000,
                step=10,
                help="Block population"
            )
            average_occupancy = st.slider(
                "Average Occupancy",
                0.5, 10.0, 3.0, 0.1,
                help="Average number of people per household",
                format="%.2f"
            )
            latitude = st.number_input(
                "Latitude",
                min_value=32.0,
                max_value=42.0,
                value=37.0,
                step=0.01,
                help="Block latitude",
                format="%.2f"
            )
            longitude = st.number_input(
                "Longitude",
                min_value=-125.0,
                max_value=-114.0,
                value=-122.0,
                step=0.01,
                help="Block longitude",
                format="%.2f"
            )

            st.session_state["IS_JSON_FILE_AVAILABLE"] = False
            st.session_state["MANUAL_INPUT"] = True

        else:  # JSON File Upload
            # Take JSON file as input
            test_input_file = st.file_uploader('Upload test prediction file', type=['json'])

            # Check if client has provided input test file
            if test_input_file:
                # Quick preview functionality for JSON input file
                st.write('Preview file')
                test_input_data = json.load(test_input_file)
                st.json(test_input_data)
                # Session is necessary, because the sidebar context acts within a
                # scope, so to access information outside the scope
                # we need to save the information into a session variable
                st.session_state["IS_JSON_FILE_AVAILABLE"] = True
                st.session_state["MANUAL_INPUT"] = False
                st.session_state["test_input_data"] = test_input_data
            else:
                # If user adds file, then performs prediction and then removes
                # file, the session var should revert back since file
                # is not available
                st.session_state["IS_JSON_FILE_AVAILABLE"] = False
                st.session_state["MANUAL_INPUT"] = False

        # Predict button
        predict_button = st.button('Predict House Price', type="primary")

    # Dashboard body
    # Heading for the dashboard
    st.write("# California Housing Price Prediction! 🏠")

    st.markdown("""
    This application predicts median house values in California based on various features.

    **Features used for prediction:**
    - Median Income (in $10,000s)
    - House Age (years)
    - Average Rooms per household
    - Average Bedrooms per household
    - Population
    - Average Occupancy (people per household)
    - Geographic Location (Latitude & Longitude)
    """)

    # If predict button is pressed
    if predict_button:
        # Prepare input data based on method
        if st.session_state.get("MANUAL_INPUT", False):
            # Manual input mode
            client_input_dict = {
                "median_income": median_income,
                "median_house_age": median_house_age,
                "average_rooms": average_rooms,
                "average_bedrooms": average_bedrooms,
                "population": float(population),
                "average_occupancy": average_occupancy,
                "latitude": latitude,
                "longitude": longitude
            }

            # Show input summary
            st.subheader("Input Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Median Income", f"${median_income * 10000:,.0f}")
                st.metric("House Age", f"{median_house_age:.0f} years")
                st.metric("Average Rooms", f"{average_rooms:.2f}")
                st.metric("Average Bedrooms", f"{average_bedrooms:.2f}")
            with col2:
                st.metric("Population", f"{population:,}")
                st.metric("Average Occupancy", f"{average_occupancy:.2f}")
                st.metric("Latitude", f"{latitude:.2f}")
                st.metric("Longitude", f"{longitude:.2f}")

            client_input = json.dumps(client_input_dict)

        elif st.session_state.get("IS_JSON_FILE_AVAILABLE", False):
            # JSON file input mode
            test_input_data = st.session_state.get("test_input_data", {})
            client_input = json.dumps(test_input_data.get('input_test', {}))
        else:
            st.error("Please either use manual input or upload a JSON file")
            st.stop()

        # Check if california_housing_model.pkl is in FastAPI folder
        if FASTAPI_HOUSING_MODEL_LOCATION.is_file():
            try:
                # This holds the result. Acts like a placeholder
                # that we can fill and empty as required
                result_container = st.empty()
                # While the model predicts show a spinner indicating model is
                # running the prediction
                with st.spinner('Predicting house price...'):
                    # Send post request to backend predict endpoint
                    predict_housing_response = requests.post(
                        f'{FASTAPI_BACKEND_ENDPOINT}/predict',
                        client_input,
                        headers={'Content-Type': 'application/json'}
                    )

                # If prediction status OK
                if predict_housing_response.status_code == 200:
                    # Convert response from JSON to dictionary
                    housing_content = json.loads(predict_housing_response.content)

                    # Display prediction result
                    st.success("Prediction Complete! ✅")

                    # Extract predicted value
                    if "predicted_value" in housing_content:
                        predicted_value = housing_content["predicted_value"]
                        predicted_price = predicted_value * 100000

                        # Display in a nice format
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Predicted Median House Value",
                                f"${predicted_price:,.2f}",
                                help="Predicted median house value in USD"
                            )
                        with col2:
                            st.metric(
                                "Value in units",
                                f"{predicted_value:.4f}",
                                help="Predicted value in $100,000 units"
                            )

                        # Additional info
                        st.info(f"""
                        **Prediction Summary:**
                        - The predicted median house value is **${predicted_price:,.2f}**
                        - This represents a value of {predicted_value:.4f} in units of $100,000
                        """)

                    elif "response" in housing_content:
                        # Alternative response format
                        predicted_value = housing_content["response"]
                        predicted_price = predicted_value * 100000
                        result_container.success(
                            f"Predicted Median House Value: ${predicted_price:,.2f}"
                        )
                    else:
                        result_container.error("Unexpected response format from server")
                        LOGGER.error("Unexpected response format")

                else:
                    # Pop up notification at bottom-left if backend is down
                    st.toast(
                        f':red[Status from server: {predict_housing_response.status_code}. '
                        f'Refresh page and check backend status]',
                        icon="🔴"
                    )
                    st.error(f"Server returned status code: {predict_housing_response.status_code}")

            except Exception as e:
                # Pop up notification if backend is down
                st.toast(
                    ':red[Problem with backend. Refresh page and check backend status]',
                    icon="🔴"
                )
                st.error(f"Error: {str(e)}")
                LOGGER.error(e)
        else:
            # Message for california_housing_model.pkl not found
            LOGGER.warning(
                'california_housing_model.pkl not found in FastAPI Lab. '
                'Make sure to run train.py to get the model.'
            )
            st.toast(
                ':red[Model california_housing_model.pkl not found. '
                'Please run the train.py file in FastAPI Lab]',
                icon="🔥"
            )


if __name__ == "__main__":
    run()