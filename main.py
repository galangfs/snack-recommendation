
# Import Library dan data Preprocessing
# Import Library dan data Preprocessing

import numpy as np
import pandas as pd
import streamlit as st

from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from bokeh.palettes import Paired12, Category20c_20

from itertools import chain
from numba import jit 
from keras.models import load_model
import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default Perangkat GPU: {}'.format(tf.test.gpu_device_name()))
else:
    print("Tidak memakai GPU")
    
print("Versi Tensorflow: {}".format(tf.__version__))


# Content-based feature engineering
# Content-based feature engineering

snack_df = pd.read_csv("snacks_data.csv")

type_initial = snack_df['type_of_food'].map(lambda x: x.split(',')).values.tolist()
all_food_types = list(set(chain(*type_initial)))

#all_food_genres
genre_initial = snack_df['food_genre'].map(lambda x: x.replace(" ", "").split(',')).values.tolist()
all_food_genres = list(set(chain(*genre_initial)))

def invert_dict(d):
    return {value: key for key, value in d.items()}

all_food_genres = sorted(list(all_food_genres)) #Mengonversi menjadi list agar dapat diurutkan berdasarkan abjad
ngenres = len(all_food_genres)

idx2genre = dict(enumerate(all_food_genres)) # Membuat Dictionary pemetaan dari indeks ke dict
genre2idx = invert_dict(idx2genre) # Inverse dict

all_food_types = sorted(list(all_food_types)) # Mengkonversi menjadi list agar dapat diurutkan berdasarkan abjad
ntypes = len(all_food_types)

idx2type = dict(enumerate(all_food_types)) # Membuat Dictionary pemetaan dari indeks ke dict
type2idx = invert_dict(idx2type) # Inverse dict

# Copy kolomnya
snack_df['features_food_genre'] = snack_df['food_genre']

# Mengubah None menjada string kosong
snack_df['features_food_genre'] = snack_df['features_food_genre'].fillna('') 

# Memisah genres menjadi list string
snack_df['features_food_genre'] = snack_df['features_food_genre'].map(lambda x: x.replace(" ", "").split(','))

#Encode genres food
def encode_genres(genres):
    out = np.zeros(ngenres)
    for genre in genres:
        if genre == '':
            pass
        else:
            out[genre2idx[genre]] = 1
    return out.tolist()

snack_df['features_food_genre'] = snack_df['features_food_genre'].map(encode_genres)

# Kopi kolomnya
snack_df['features_food_type'] = snack_df['type_of_food']

# Mengubah None menjadi string kosong
snack_df['features_food_type'] = snack_df['features_food_type'].fillna('') 

# Memisah genres menjadi list string
snack_df['features_food_type'] = snack_df['features_food_type'].map(lambda x: x.split(','))


def encode_type(types):
    out = np.zeros(ntypes)
    for type in types:
        if type == '':
            pass
        else:
            out[type2idx[type]] = 1
    return out.tolist()

snack_df['features_food_type'] = snack_df['features_food_type'].map(encode_type)

#  **Collaborative-filtering feature engineering**
#  **Collaborative-filtering feature engineering**

# Cek struktur data
rating = pd.read_csv("rating_snacks.csv")

rating = rating[rating['food_id'].isin(snack_df['food_id'])] 

rating['rating'].replace({-1: np.nan}).dropna()

user_median = rating.groupby('user_id').median()['rating']
overall_median = user_median.median()
user_median = dict(user_median.replace({-1 : overall_median}))
user_medians = rating['user_id'].apply(lambda x: user_median[x])
rating['rating'] = rating['rating'].replace({-1 : np.nan}).fillna(user_medians)
rating['rating'] = rating['rating'] / rating['rating'].max() # Normalisasi dengan membagi yang terbesar

num_neg = 4
user2n_snack = dict(rating.groupby('user_id').count()['food_id'])

#unique user ids
all_users = np.sort(rating['user_id'].unique())
#unique food ids
all_snacks = np.sort(rating['food_id'].unique())
n_snacks = len(all_snacks)
n_users = len(all_users)

@jit
def choice_w_exclusions(array, exclude, samples):
    max_samples = len(array)-len(exclude)
    final_samples = min(samples, max_samples)
    possible = np.array(list(set(array) - set(exclude)))
    np.random.seed(0)
    return np.random.choice(possible, size = final_samples, replace = False)
@jit
def flat(l):
    return [item for sublist in l for item in sublist]


# ### Sample negative entries
neg_user_id = []
neg_snack_id = []
neg_rating = []

for user in all_users:
    #Pengecualian food ids untuk user id spesifik
    exclude = list(rating[rating['user_id'] == user]['food_id'])
    sampled_snack_id = choice_w_exclusions(all_snacks, exclude, len(exclude) * num_neg)
    
    neg_user_id.append([user] * len(sampled_snack_id))
    neg_snack_id.append(sampled_snack_id)
    neg_rating.append([0.] * len(sampled_snack_id))
    
neg_user_id = flat(neg_user_id)
neg_snack_id = flat(neg_snack_id)
neg_rating = flat(neg_rating)


negatives = pd.DataFrame({'user_id': neg_user_id,
                          'food_id': neg_snack_id,
                          'rating': neg_rating})
data = pd.concat([rating, negatives], ignore_index = True)



snack_df['features'] = snack_df['features_food_genre'] + snack_df['features_food_type']
snack_df['features'] = snack_df['features'].apply(np.array)

n_feats = len(snack_df['features'].iloc[0])

data = data.join(snack_df['features'], on = 'food_id').dropna()


#Membuat dictionary untuk menghubungkan snack id ke item id
snack2item_dict = dict(zip(np.sort(all_snacks), list(range(n_snacks))))
item2snack_dict = {v: k for k, v in snack2item_dict.items()}

def snack2item(s_id):
    return snack2item_dict[s_id]

def item2snack(i_id):
    return item2snack_dict[i_id]
                       
data['item_id'] = data['food_id'].apply(snack2item)


# Untuk menggunakan kembali model di lain waktu untuk membuat prediksi, kami memuat model yang disimpan
# Untuk menggunakan kembali model di lain waktu untuk membuat prediksi, kami memuat model yang disimpan

model = load_model("static/model.h5")

# Membuat modul explore 
# Membuat modul explore 

indexed_snacks = snack_df.set_index('food_id')
def explore(user_id):
    sub = rating[rating['user_id'] == user_id]

    bought_snacks = sub['food_id']
    
    ratings = sub['rating']
    
    names = indexed_snacks.loc[bought_snacks]['food_name']
    
    genres = indexed_snacks.loc[bought_snacks]['food_genre']
    
    types = indexed_snacks.loc[bought_snacks]['type_of_food']
    
    rating_info = pd.DataFrame(zip(bought_snacks, names,
                                   genres, types,ratings*10),
                               columns = ['food_id', 'name',
                                          'genre','type_of_food', 'rating']).set_index('food_id')
    return rating_info.sort_values(by = 'rating', ascending = False).iloc[:]

# Membuat modul recommend 
def recommend(user_id):
    # makanan ringan yang telah dibeli oleh pembeli
    bought_snacks = np.sort(rating[rating['user_id'] == user_id]['food_id'])
    
    # semua makanan yang tidak dibeli oleh pembeli
    test_snacks = np.array(list(set(all_snacks) - set(bought_snacks)))
    
    # membuat array id users dengan panjang yang sama dengan semua makanan ringan yang tidak dibeli oleh user
    test_user = np.array([user_id] * len(test_snacks))
    
    # membuat berbagai makanan ringan yang tidak dibeli dengan id item masing-masing
    test_items = np.array([snack2item(a) for a in test_snacks])
    
    #untuk makanan ringan yang tidak dibeli oleh user 
    # semua kolom(food_id,food_name,type of food,food_genre,hot encoded feature column)
    sub_snack = indexed_snacks.loc[test_snacks]
    
    #stacking the food_genre features columns
    test_features = np.stack(sub_snack['features'].to_numpy())
    
    test = [test_user, test_items, test_features]
    preds = model.predict(test).flatten()
    
    results = pd.DataFrame(zip(sub_snack['food_name'], test_snacks,  sub_snack['food_genre'],sub_snack['type_of_food'], preds * 10),
                           columns = ['name', 'food_id',
                                      'genre','type_of_food', 'score']).set_index('food_id')
    return results.sort_values(by = 'score', ascending = False).iloc[:]



# --- Pembuatan modul Streamlit --- 
# --- Pembuatan modul Streamlit --- 

'''
# Sistem Rekomendasi Pemfilteran Kolaboratif Neural

'''

user_id = st.number_input("Hai, Masukkan user_id 0-10093 untuk mencoba eksplor",value=8177, min_value=0, max_value=10093)
explore_df = recommend(user_id)
recommend_df = explore(user_id)

usr_headline = "## Aktifitas dari Pembeli - " + str(user_id) + "\n" + " Berbagai Tren dan statistik dilakukan oleh pembeli hingga saat ini."
st.markdown(usr_headline)

st.subheader("Dataframe Aktifitas Pembeli - ")
st.dataframe(explore_df)


st.subheader("Tipe Makanan yang dibeli oleh Pembeli -")
explore_types_of_food = pd.DataFrame(explore_df.groupby('type_of_food')['type_of_food'].count())
explore_color_type = list(Paired12[:len(explore_types_of_food)])
fig_explore_types_of_food = go.Figure(data=[go.Bar(x=explore_types_of_food.index, y=explore_types_of_food["type_of_food"], marker_color=explore_color_type)])
fig_explore_types_of_food.update_layout(title="Tipe Makan Ringan dan Jumlah Produk", xaxis_title='Tipe Makanan Ringan', yaxis_title='Jumlah Produk', xaxis_tickangle=-45, plot_bgcolor="#707070",width=800, height=600)    
st.plotly_chart(fig_explore_types_of_food)

st.subheader("Tipe Genre yang dibeli oleh Pembeli -")
explore_list_of_genre = explore_df.genre.str.split(',').map(lambda colvalue : [s.strip() for s in colvalue])
explore_genre_count = dict(Counter(x for xs in explore_list_of_genre for x in set(xs)))
explore_genre_counts = pd.DataFrame(explore_genre_count.items(), columns=['genre', 'count'])
explore_color_genre = list(Category20c_20[:len(explore_genre_counts)])

fig_explore_genre_counts = go.Figure(data=[go.Bar(x=explore_genre_counts['count'], y=explore_genre_counts["genre"], orientation='h', marker_color=explore_color_genre)])
fig_explore_genre_counts.update_layout(title="Tipe Genre dan Jumlah Produk", xaxis_title='Jumlah Produk', yaxis_title='Genre', plot_bgcolor="#707070",width=800, height=600)    
st.plotly_chart(fig_explore_genre_counts)


usr_recommended_headline = "Aktifitas dari id pembeli - " + str(user_id) + " berdasarkan Rekomendasi \n" + "Berbagai Tren dan statistik direkomendasikan untuk pembeli"
st.markdown(usr_recommended_headline)

st.subheader("Dataframe Rekomendasi dari Pembeli -")
st.dataframe(recommend_df)


st.subheader("Tipe Makanan Ringan yang Direkomendasikan -")
recommend_types_of_food = pd.DataFrame(recommend_df.groupby('type_of_food')['type_of_food'].count())
recommend_color_type = list(Paired12[:len(recommend_types_of_food)])
fig_recommend_types_of_food = go.Figure(data=[go.Bar(x=recommend_types_of_food.index, y=recommend_types_of_food["type_of_food"], marker_color=recommend_color_type)])
fig_recommend_types_of_food.update_layout(title="Tipe Makananan Ringan dan Jumlah Produk", xaxis_title='Tipe Makanan Ringan', yaxis_title='Jumlah Produk', xaxis_tickangle=-45,plot_bgcolor="#707070",width=800, height=600)    
st.plotly_chart(fig_recommend_types_of_food)

st.subheader("Tipe Genre yang Direkomendasikan -")
recommend_list_of_genre = recommend_df.genre.str.split(',').map(lambda colvalue : [s.strip() for s in colvalue])
recommend_genre_count = dict(Counter(x for xs in recommend_list_of_genre for x in set(xs)))
recommend_genre_counts = pd.DataFrame(recommend_genre_count.items(), columns=['genre', 'count'])
recommend_color_genre = list(Category20c_20[:len(recommend_genre_counts)])

fig_recommend_genre_counts = go.Figure(data=[go.Bar(x=recommend_genre_counts['count'], y=recommend_genre_counts["genre"], orientation='h', marker_color=recommend_color_genre)])
fig_recommend_genre_counts.update_layout(title="Tipe Genre dan Jumlah Produk", xaxis_title='Jumlah Produk', yaxis_title='Genre',plot_bgcolor="#707070",width=800, height=600)    
st.plotly_chart(fig_recommend_genre_counts)

