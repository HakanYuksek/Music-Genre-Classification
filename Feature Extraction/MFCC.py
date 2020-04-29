import numpy as np
import sqlite3
import librosa
import librosa.display
import os

con = sqlite3.connect("Project.db")
cursor = con.cursor()

rowName=["VarMFCC1","VarMFCC2","VarMFCC3","VarMFCC4","VarMFCC5","VarMFCC6"
        ,"VarMFCC7","VarMFCC8","VarMFCC9","VarMFCC10","VarMFCC11","VarMFCC12"
         ,"VarMFCC13","VarMFCC14","VarMFCC15","VarMFCC16","VarMFCC17","VarMFCC18"
         ,"VarMFCC19","VarMFCC20","DMFCC1","VarDMFCC1","DMFCC2","VarDMFCC2"
         ,"DMFCC3","VarDMFCC3","DMFCC4","VarDMFCC4","DMFCC5","VarDMFCC5"
         ,"DMFCC6","VarDMFCC6","DMFCC7","VarDMFCC7","DMFCC8","VarDMFCC8"
         ,"DMFCC9","VarDMFCC9","DMFCC10","VarDMFCC10","DMFCC11","VarDMFCC11"
         ,"DMFCC12","VarDMFCC12","DMFCC13","VarDMFCC13","DMFCC14","VarDMFCC14"
         ,"DMFCC15","VarDMFCC15","DMFCC16","VarDMFCC16","DMFCC17","VarDMFCC17"
         ,"DMFCC18","VarDMFCC18","DMFCC19","VarDMFCC19","DMFCC20","VarDMFCC20"]


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
        #calculate and add mfcc
        tmp=librosa.feature.mfcc(y,sr)
        tmpp=[None]*20
        n_mfcc=20
        mfcc_col_size = len(tmp[0])
        for i in range(20):
            tmpp[i]=np.var(tmp[i])
        features=np.append(features,tmpp)
        
        #MFCC derivative feature cikarimi yapilir
        mfcc_derivative = np.empty([n_mfcc, mfcc_col_size], dtype=float)
        #Ilk sutunlar 0 olarak ilklendirilir
        for i in range(n_mfcc):
            mfcc_derivative[i][0] = 0
        #Turevler hesaplanir ve matris doldurulur
        for i in range(n_mfcc):
            for j in range(1, mfcc_col_size):  
                mfcc_derivative[i][j] = tmp[i][j] - tmp[i][j-1]
               
                
        #Ortalama alinip features'a eklenecek
        mfcc_derivative_avgs_stds = np.empty(2*n_mfcc, dtype=float)
        j=0  #mfcc_derivative_avgs_stds dizisinin indisi
        for i in range(n_mfcc):
            #Ilk degerler 0 ortalamada buna dikkat!
            mfcc_derivative_avgs_stds[j] = np.average(mfcc_derivative[i])  
            mfcc_derivative_avgs_stds[j+1] = np.var(mfcc_derivative[i])
            j = j+2
        features = np.append(features, mfcc_derivative_avgs_stds)
        
        t=0
        print(features)
        name="'"+name+"'"
        while(t<len(rowName)):
            cursor.execute("UPDATE Features SET "+rowName[t]+" = "+str(features[t])+" WHERE Name ="+name)
            con.commit()               
            t=t+1
        print("added...")
