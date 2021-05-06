import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from sklearn.preprocessing import LabelEncoder

from sklearn.datasets import load_breast_cancer

import pandas as pd
dff = pd.read_csv('dataMalangE.csv')
dff.dropna(inplace=True)

values = dff.values
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
d = pd.DataFrame(values)
d.columns = ['Tanggal','Temperatur','Presipitasi','Kec.Angin','Cuaca','ArahAngin','JarakPandang','Awan','Kelembapan']
d.to_csv('dataMalangE.csv',index=False)
df = pd.DataFrame(dff)

print(df.shape)

import seaborn as sns

correlation_mat = df.corr()

sns.heatmap(correlation_mat, annot = True)

plt.show()