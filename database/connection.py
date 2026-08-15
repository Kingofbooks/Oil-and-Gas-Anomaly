from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

class ConnectionDB:
    def __init__(self):
        self.DATABASE_URL=os.getenv("DATABASE_URL")
    
    def connect_test(self):
        engine = create_engine(self.DATABASE_URL)
        with engine.connect() as connection:
            print("Connected to PostgreSQL!")
        
def main():
    conn=ConnectionDB()
    conn.connect_test()

if __name__=="__main__":
    main()