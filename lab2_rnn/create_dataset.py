# 1st STEP --> music dataset
# Bach dataset with 62 chorales, total of 5665 samples with different times,
# converted into npz of (47 songs x 70 events of time x 17 attributes per song)
# Olga Valls
#
import pandas as pd
import numpy as np
from sklearn import preprocessing

df = pd.read_csv("datasets/bach.csv", names=["id_choral", "id_event", "noteC", "noteC#", "noteD", "noteD#", "noteE", "noteF", "noteF#", "noteG", "noteG#", "noteA", "noteA#", "noteB", "bass", "metter", "chord"])

# de quin tipus es cada columna
# print(df.dtypes)

# les que vull passar d'object a categorical
df['id_choral'] = df['id_choral'].astype('category')
df['noteC'] = df['noteC'].astype('category')
df['noteC#'] = df['noteC#'].astype('category')
df['noteD'] = df['noteD'].astype('category')
df['noteD#'] = df['noteD#'].astype('category')
df['noteE'] = df['noteE'].astype('category')
df['noteF'] = df['noteF'].astype('category')
df['noteF#'] = df['noteF#'].astype('category')
df['noteG'] = df['noteG'].astype('category')
df['noteG#'] = df['noteG#'].astype('category')
df['noteA'] = df['noteA'].astype('category')
df['noteA#'] = df['noteA#'].astype('category')
df['noteB'] = df['noteB'].astype('category')
df['bass'] = df['bass'].astype('category')
df['chord'] = df['chord'].astype('category')
# print('\n')
# print(df.dtypes)

# columnes que són de tipus categorical
cat_columns = df.select_dtypes(['category']).columns
# print(cat_columns)

# assigno codis a les que he passat a categorical
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
# print(df)

# Ara ja ho tinc tot en números, tot i que no sé com ha fet les categories...

# les columnes que normalitzo. La resta no cal pq. són 0 o 1 (notes)
list_norm = ["id_choral", "id_event", "bass", "metter", "chord"]

# for col in df.columns:  # go through all of the columns
#     if col in list_norm:  # normalize all ... except for the target itself!
#         # df[col] = df[col].pct_change()
#         # Percentage change between the current and a prior element.
#         # Computes the percentage change from the immediately previous row by default. This is useful in comparing the percentage of change in a time series of elements.
#         # df.dropna(inplace=True)  # remove the nas created by pct_change
#         df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.
# df.dropna(inplace=True)  # cleanup again... jic. Those nasty NaNs love to creep in.
# print(df)

# Agrupo per choral (cançó)
dict_chorales = {k: v for k, v in df.groupby('id_choral')}

# A final creo llista (num samples >=30 sequences) de matrius sequences x attributes
# les cançons que no tinguin mínim 30 events no les agafo
rows = 70
final = []
count = 0

for k in dict_chorales:
    temp = np.array(dict_chorales[k].values)
    # print('size temp: {}'.format(temp.shape))
    # print('temp:')
    # print(temp)
    # trec els dos primers attributes pq. no els necessitaré per a la RNN
    temp = np.delete(temp, np.s_[0:2], axis=1)
    # temp = temp / temp.max(axis=0)
    print('size temp: {}'.format(temp.shape))
    # print('temp:')
    # print(temp)
    if temp.shape[0] < rows:
        print('pass -- shape[0]: {}'.format(temp.shape[0]))
        count = count + 1
    else:
        print('shape[0]: {}'.format(temp.shape[0]))
        n = temp.shape[0] - rows
        if n != 0:
            a = temp[:-n, :]
        else:
            a = temp
        print('a shape: {}'.format(a.shape))
        final.append(a)
        print('--\nlen final: {}'.format(len(final)))

print('count out: {}'.format(count))    # les que he descartat
#print('final shape: {}\nfinal:\n{}'.format(len(final), final))
print('final shape: {}'.format(len(final)))

# # matriu numpy 3D de (samples x sequences x attributes) = (47x70x17)
np_data = np.array(final)
print('np_data: {}'.format(np_data))
print('shape np_data: {}'.format(np_data.shape))

# np_data_normed = np_data / np_data.max(axis=0)
# print('FINAL MATRIX')
# print(np_data_normed)
# print(np_data_normed.shape)

# guardo cada element de la llista (choral) com a array individual
# Save several arrays into a single file in uncompressed .npz format.
# np.savez('datasets/bach_norm.npz', info=np_data_normed)
# np.savez('datasets/bach3.npz', *final)
np.savez('datasets/bach_dataset.npz', *final)

# # check if everything is ok
container = np.load('datasets/bach_dataset.npz')
data = [container[matriu] for matriu in container]
print('matrius dins npz: {}, shape de cada matriu: {}'.format(len(data), data[0].shape))
