import matplotlib.pyplot as plt
import os,sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
file_path = PROJECT_DIR + "/resource/process_record.txt"

y_value = list()

with open(file_path,"r") as f:
    result = f.readline()
    result.strip()
    y_value = [int(item) for item in result.split('\t') if item.isnumeric()]


plt.plot(y_value)
plt.ylabel("living steps")
plt.show()







