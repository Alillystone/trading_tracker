import datetime

def current_date():
    return datetime.datetime.now()

def convert_date_to_string(date):
    return datetime.datetime.strptime(date,'%Y-%m-%d')