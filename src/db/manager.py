from sqlalchemy import create_engine, Column, Integer, String, DateTime, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

from constants import DB_URI, DB_CHATBOT

engine = create_engine(DB_URI)
Base = declarative_base()

class Character(Base):
    __tablename__ = DB_CHATBOT
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    instruction = Column(String, nullable=False)
    
def create_tables():
    Base.metadata.create_all(engine)

def check_exist_table():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE';"))
        result = [table[0] for table in result]
        
        if DB_CHATBOT in result:
            return True
        else:
            return False

def add_data(name, instruction):
    Session = sessionmaker(bind=engine)
    session = Session()
    new_character = Character(
        name=name,
        instruction=instruction,
    )
    session.add(new_character)
    session.commit()
    session.close()
    
def get_all_data():
    Session = sessionmaker(bind=engine)
    session = Session()
    
    data = session.query(Character).all()
    data = [
        {
            'name': character.name,
            'instruction': character.instruction,
        }
        for character in data
    ]
    session.close()
    return data

def update_data_by_name(name, instruction=None):
    Session = sessionmaker(bind=engine)
    session = Session()
    
    if not name:
        session.close()
        return
    
    character = session.query(Character).filter_by(name=name).first()
    if character:
        if name:
            character.name = name
        if instruction:
            character.instruction = instruction
        session.commit()
    
    session.close()
    
def delete_data_by_name(name):
    Session = sessionmaker(bind=engine)
    session = Session()
    
    if not name:
        session.close()
        return
    
    character = session.query(Character).filter_by(name=name).first()
    if character:
        session.delete(character)
        session.commit()
    
    session.close()
    
def delete_all_data():
    Session = sessionmaker(bind=engine)
    session = Session()
    
    session.query(Character).delete()
    session.commit()
    
    session.close()
    
def delete_table():
    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS "{DB_CHATBOT}";'))