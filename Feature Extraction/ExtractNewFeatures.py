import numpy as np
import sqlite3
import librosa
import librosa.display
import os

con = sqlite3.connect("Project.db")
cursor = con.cursor()

rowName=["MeanSpecFlatness","MedianSpecFlatness","VarSpecFlatness","MeanPoly","MedianPoly","VarPoly"]

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
        features=[np.average(librosa.feature.spectral_flatness(y)),np.median(librosa.feature.spectral_flatness(y)),
                  np.var(librosa.feature.spectral_flatness(y))
                  ]
        coeff=librosa.feature.poly_features(y,sr)
        tmpp=[np.average(coeff[0]),np.median(coeff[0]),np.var(coeff[0])]
        features=np.append(features,tmpp)
        t=0
        print(features)
        name="'"+name+"'"
        while(t<len(rowName)):
            cursor.execute("UPDATE Features SET "+rowName[t]+" = "+str(features[t])+" WHERE Name ="+name)
            con.commit()               
            t=t+1
        print("added...")
