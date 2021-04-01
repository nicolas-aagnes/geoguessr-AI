import os

with open("generator/countries.txt", "r") as f:
    for country in f.readlines():
        print(country.strip())
        os.mkdir(f"data/{country.strip()}")
