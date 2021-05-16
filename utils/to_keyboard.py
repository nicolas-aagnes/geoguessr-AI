import argparse
import time

import keyboard

parser = argparse.ArgumentParser()
parser.add_argument("--file", required=True)
args = parser.parse_args()

with open(args.file, "r") as f:
    contents = f.readlines()

print(contents)

time.sleep(3)
keyboard.write("".join(contents))
