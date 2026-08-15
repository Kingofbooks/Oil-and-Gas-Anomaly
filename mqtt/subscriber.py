import paho.mqtt.client as mqtt

class MqttSubscriber:
    def __init__(self):
        self.client=mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_message = self.on_message
        self.client.connect("localhost",1883)
        self.client.subscribe("oilgas/WELL-001/sensors")
    
    def on_message(self, client, userdata, message):
        print("Received:")
        print("Topic:", message.topic)
        print("Message:", message.payload.decode())


def main():
    sub=MqttSubscriber()
    sub.client.loop_forever()


if __name__ == "__main__":
    main()