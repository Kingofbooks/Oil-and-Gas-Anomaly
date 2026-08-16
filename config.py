import os
from dotenv import load_dotenv

load_dotenv()


class Settings:

    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")
        self.mqtt_broker_host = os.getenv("MQTT_BROKER_HOST")
        self.mqtt_broker_port = int(os.getenv("MQTT_BROKER_PORT"))
        self.mqtt_topic = os.getenv("MQTT_TOPIC")