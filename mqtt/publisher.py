import pandas as pd
import paho.mqtt.client as mqtt
import json
import time
from pathlib import Path

class MqttPublisher:
    def __init__(self):
        self.csv_path = Path(
            r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\oil gas anomaly\data\sensor_500.csv"
        )
    def connect(self):
        client=mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.connect("localhost", 1883)
        
        return client
    
    def load_data(self):
        df = pd.read_csv(self.csv_path)
        return df

    def publish_reading(self, row):
        data = row.to_dict()
        payload = json.dumps(data)

        return payload

    def stream(self,df,client,limit=None,interval=1):
        if limit is None:
            limit=500
        
        topic="oilgas/WELL-001/sensors"
        for idx,row in df.head(limit).iterrows():
                payload=self.publish_reading(row)
                client.publish(
                    topic,
                    payload
                )
                print("Published:")
                print("Topic:", topic)
                print("Message:", payload)
                time.sleep(interval)
                
    def disconnect(self, client):
        client.disconnect()
def main():
    pub=MqttPublisher()
    try:
        client=pub.connect()
        df=pub.load_data()
        pub.stream(df,client,2,1)
    finally:
        pub.disconnect(client)
if __name__=="__main__":
    main()