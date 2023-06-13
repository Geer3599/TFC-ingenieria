import pandas as pd 

df = pd.read_excel('datoslechuzas.xlsx', sheet_name='Hoja1')
df = df.drop(['P.F.V.', 'Fecha', 'Energía kWh', 'Energía Delta kWh', 'Veloc. Viento m/s' ],axis = 1)

df.columns = ['Potencia', 'Radiacion', 'Temperatura','Temperatura panel']
df = df.fillna(0)
df = df.round(0)
print(df)

df.to_csv('datafinlechuza.csv')