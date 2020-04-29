import Project
import sqlite3
import librosa
import os

# This program inserts features from GTZAN data set to Project.db

con = sqlite3.connect("Project.db")
cursor = con.cursor()

rowName=["Name","MeanZeroCross","MedianZeroCross","VarZeroCross","MeanSpecBand","MedianSpecBand","VarSpecBand","MeanRmse","MedianRmse","VarRmse"
                 ,"MeanSpecCentroid","MedianSpecCentroid","VarSpecCentroid","MeanSpecRoll","MedianSpecRoll","VarSpecRoll","MeanSpecContrast","MedianSpecContrast"
                 ,"VarSpecContrast","CQT1","CQT2","CQT3","CQT4","CQT5","CQT6","CQT7","CQT8","CQT9","CQT10","CQT11","CQT12","MFCC1","MFCC2","MFCC3","MFCC4"
                 ,"MFCC5","MFCC6","MFCC7","MFCC8","MFCC9","MFCC10","MFCC11","MFCC12","MFCC13","MFCC14","MFCC15","MFCC16","MFCC17","MFCC18","MFCC19","MFCC20"
                 ,"MeanTonnetz1","MedianTonnetz1","VarTonnetz1","MeanTonnetz2","MedianTonnetz2","VarTonnetz2","MeanTonnetz3","MedianTonnetz3","VarTonnetz3"
                 ,"MeanTonnetz4","MedianTonnetz4","VarTonnetz4","MeanTonnetz5","MedianTonnetz5","VarTonnetz5","MeanTonnetz6","MedianTonnetz6","VarTonnetz6"
        ]

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
        sleep(100)
        features = Project.construct_feature_list(name,musicData,sr)
        str1=rowName[0]
        t=1
        while(t<len(rowName)):
            str1=str1+","+rowName[t]
            t=t+1
        m="?"
        for l in range(len(features)-1):
            m=m+",?"
        cursor.execute("INSERT INTO Features ("+str1+") VALUES("+m+")",(features))
        con.commit()
        genre=i
        genre="'"+genre+"'"
        name="'"+name+"'"
        cursor.execute("UPDATE Features SET Genre = "+genre+" WHERE Name="+name)
        con.commit()
        print("added...")
