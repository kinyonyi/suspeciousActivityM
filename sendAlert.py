#pip install python-dotenv
from dotenv import load_dotenv
import os
from twilio.rest import Client

load_dotenv()

# Access environment variables
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
from_number = os.getenv("TWILIO_FROM_NUMBER")
to_number = os.getenv("TWILIO_TO_NUMBER")

def send_alert(alertType):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    message = client.messages.create(
        body=f"{alertType} detected! Immediate attention required.",
        from_=TWILIO_PHONE_NUMBER,
        to=EMERGENCY_CONTACT
    )
    res = "Alert Sent: ", message.sid
    print(res)
    return res
