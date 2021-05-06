import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

dff = pd.read_csv('dataMalangOE.csv')
data = pd.read_csv('dataMalangO.csv', parse_dates=True)

def scatterplot(data):
    pio.renderers.default="browser"
    data.dropna(inplace=True)
    x = data.Temperature
    y = data.HeatIndex
    z = data.Precipitation
    c = data.Conditions
    df = pd.DataFrame({
    'cat':c, 'Temperature':x, 'HeatIndex':y, 'Precipitation':z
    })
    df.head()
    fig = px.scatter_3d(df, x='Temperature', y='HeatIndex', z='Precipitation',
                        color='cat',
                        title="3D Scatter Plot")
    fig.show()


# def correlationplot(dff):
dff.dropna(inplace=True)
cols = list(dff)
d = pd.DataFrame(dff)
encoder = LabelEncoder()
values = d.values
values[:,11] = encoder.fit_transform(values[:,11])
df = pd.DataFrame(values)
df.columns = cols
# df.to_csv('dataMalangOE.csv', index=False)
print(df.shape)
correlation_mat = dff.corr()
sns.heatmap(correlation_mat, vmin=-1, vmax=1, annot=True)
plt.show()
    
dff = pd.read_csv('dataMalangO.csv')
data = pd.read_csv('dataMalangO.csv', parse_dates=True)

scatterplot(data)
# correlationplot(dff)
