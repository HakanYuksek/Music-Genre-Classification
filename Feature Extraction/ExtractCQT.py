import numpy as np
import sqlite3
import librosa
import librosa.display
import os

con = sqlite3.connect("Project.db")
cursor = con.cursor()

rowName=["VarCQT1","VarCQT2","VarCQT3","VarCQT4","VarCQT5","VarCQT6","VarCQT7"
        ,"VarCQT8","VarCQT9","VarCQT10","VarCQT11","VarCQT12"]

path="\\genres"
GenreList=os.listdir("\\genres")    

for i in GenreList:
    filePath=path + "\\" + i
    musicFiles=os.listdir(filePath)
    for j in musicFiles:
        musicFileName=filePath + "\\" + j
        print(musicFileName)
        musicData,sr=librosa.load(musicFileName)
        mname=j.split(".")
        name=""
        for j in range(len(mname)-1):
            name=name +mname[j]
        print(name)
        y=musicData
        features=[]
        #calculate and add var chroma_cqt
        tmp=librosa.feature.chroma_cqt(y,sr)
        tmpp=[None]*12
        for i in range(12):
            tmpp[i]=np.var(tmp[i])
        features=np.append(features,tmpp)
        t=0
        print(features)
        name="'"+name+"'"
        while(t<len(rowName)):
            cursor.execute("UPDATE Features SET "+rowName[t]+" = "+str(features[t])+" WHERE Name ="+name)
            con.commit()               
            t=t+1
        print("added...")
