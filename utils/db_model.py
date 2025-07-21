# === utils/db_model.py ===
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()
engine = create_engine('sqlite:///attendance.db')
Session = sessionmaker(bind=engine)
session = Session()

class Attendance(Base):
    __tablename__ = 'attendance'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    date = Column(String)
    check_in_time = Column(String)
    check_out_time = Column(String)
    image_path = Column(String)

Base.metadata.create_all(engine)

# === app.py ===
from flask import Flask, render_template
from utils.db_model import session, Attendance

app = Flask(__name__)

@app.route('/')
def index():
    records = session.query(Attendance).order_by(Attendance.id.desc()).all()
    return render_template('index.html', records=records)

if __name__ == '__main__':
    app.run(debug=True)