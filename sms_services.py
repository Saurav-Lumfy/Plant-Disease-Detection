# sms_services.py (new file)
from twilio.rest import Client
import streamlit as st

def send_verification_sms(phone_number, otp):
    """Send SMS with verification code using Twilio"""
    try:
        client = Client(
            st.secrets["twilio"]["account_sid"],
            st.secrets["twilio"]["auth_token"]
        )
        
        message = client.messages.create(
            body=f"Your AgroDoc verification code: {otp}",
            from_=st.secrets["twilio"]["phone_number"],
            to=phone_number
        )
        return message.sid
    except Exception as e:
        st.error(f"Failed to send SMS: {str(e)}")
        return None