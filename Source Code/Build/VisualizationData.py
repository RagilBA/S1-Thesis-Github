import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

dff = pd.read_csv('dataMalang-Modified.csv')
data = pd.read_csv('dataMalang-Modified.csv', parse_dates=True)

def scatterplot(data):
    pio.renderers.default="browser"
    data.dropna(inplace=True)
    x = data.Precipitation
    y = data.CloudCover
    z = data.Temperature
    c = data.Conditions
    df = pd.DataFrame({
    'cat':c, 'Precipitation':x, 'CloudCover':y, 'Temperature':z
    })
    df.head()
    fig = px.scatter_3d(df, x='Precipitation', y='CloudCover', z='Temperature',
                        color='cat',
                        title="3D Scatter Plot")
    fig.show()


# def correlationplot(dff):
dff.dropna(inplace=True)
cols = list(dff)
d = pd.DataFrame(dff)
encoder = LabelEncoder()
values = d.values
values[:,9] = encoder.fit_transform(values[:,9])
df = pd.DataFrame(values)
df.columns = cols
# df.to_csv('dataMalangOE.csv', index=False)
print(df.shape)
correlation_mat = dff.corr()
sns.heatmap(correlation_mat, vmin=-1, vmax=1, annot=True)
plt.show()
    
# dff = pd.read_csv('dataMalangO.csv')
# data = pd.read_csv('dataMalangO.csv', parse_dates=True)

scatterplot(data)
# correlationplot(dff)
