"""This module contains an ETL pipeline
which processes the GDP data obtained from the
World Bank website.

ETL pipelines (i.e., Extract, Transform, Load)
are common before applying any data analysis or
modeling. In the present case, these steps are
carried out:

1. 
2. 
3.

Author: Mikel Sagardia
Date: 2023-03-06
"""
import sqlite3
import numpy as np
import pandas as pd

SOURCE_FILENAME = "../10_imputation/gdp_data.csv"
DB_FILENAME = "world_bank.db"

def create_database_table():
    """Create the database file with the gdp table."""
    # Connect to the database
    # sqlite3 will create this database file if it does not exist already
    conn = sqlite3.connect(DB_FILENAME)

    # Get a cursor
    cur = conn.cursor()

    # Drop the gdp table in case it already exists
    cur.execute("DROP TABLE IF EXISTS gdp")

    # Create the gdp table: long format, with these rows:
    # countryname, countrycode, year, gdp
    cur.execute("CREATE TABLE gdp (countryname TEXT, countrycode TEXT, year INTEGER, gdp REAL, PRIMARY KEY (countrycode, year));")

    # Commit and close
    conn.commit()
    conn.close()

# Generator for reading in one line at a time
# generators are useful for data sets that are too large to fit in RAM
# You do not need to change anything in this code cell
def extract_lines(file):
    """Generator for reading"""
    while True:
        line = file.readline()
        if not line:
            break
        yield line

if __name__ == "__main__":
    pass