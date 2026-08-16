from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
from database.models import Base
from config import Settings
load_dotenv()

class ConnectionDB:
    def __init__(self):
        self.setting=Settings()
        self.database_url=self.setting.database_url
        self.engine = create_engine(self.database_url)
        self.SessionLocal=sessionmaker(bind=self.engine)
        
    def connect_test(self):
        with self.engine.connect():
            print("Connected to PostgreSQL!")
    
    def create_tables(self):
        Base.metadata.create_all(self.engine)
        print("Tables created!")
def main():
    conn=ConnectionDB()
    conn.connect_test()
    conn.create_tables()

if __name__=="__main__":
    main()