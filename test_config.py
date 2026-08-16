from config import Settings


def main():

    settings = Settings()

    print("Broker:", settings.mqtt_broker_host)
    print("Port:", settings.mqtt_broker_port)
    print("Topic:", settings.mqtt_topic)


if __name__ == "__main__":
    main()