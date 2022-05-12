import os
import time

def main():
    os.system("python setup.py install")
    os.system("pip install seaborn")
    os.system("pip install selenium==3.8.0")
    os.system("conda install phantomjs -c conda-forge")

if __name__ == '__main__':
    main()