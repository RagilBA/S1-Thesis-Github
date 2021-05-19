Program klasifikasi pada file data belum di update semua, apabila ingin digunakan maka tambahkan line ini:

##Tambahkan setelah baris *testY = pd.read_csv('TestConditions.csv')*##
data_dates = pd.to_datetime(testY['Date'])

##Tambahkan pada baris baling akhir##
PdYact = pd.DataFrame(Yact)
PdYpred = pd.DataFrame(Ypred)
PdCon = pd.concat([PdYact, PdYpred], axis=1, join='inner')
Pd = pd.concat([data_dates,PdCon], axis =1)
Pd.columns = ['Date', 'Actual Conditions','Predict Conditions']
Pd.to_csv('Conditions Comparison.csv', index = False)