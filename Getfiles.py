import wget
import pandas as pd
import os
# filenames= "https://cs7ns1.scss.tcd.ie/?shortname=janaparn"
# locationnames= r"./"
# retries = 1
# success = False
# while not success:
#     try:
#         wget.download(filenames, out=locationnames)
#         # print("downloaded=",row[0])
#         success = True
#         print("Success with the filenames csv")
#     except Exception as e:
#         print('Error while downloading filenames.csv retrying',retries)
#         retries += 1

url = "https://cs7ns1.scss.tcd.ie"
shortname = "janaparn"
location= r"./test"
if not os.path.exists(location):
        print("Creating output directory " + location)
        os.makedirs(location)
df = pd.read_csv("janaparn-challenge-filenames.csv",header=None)

for index, row in df.iterrows():
    retries = 1
    success = False
    # print(row[0])
    entireurl = url+"/?shortname="+shortname+"&myfilename="+row[0]
    print(entireurl)
    while not success:
        try:
            wget.download(entireurl, out=location)
            # print("downloaded=",row[0])
            success = True
            print("Success with ",row[0])
        except Exception as e:
            print(e)
            print('Error! retrying',retries)
            retries += 1
    


