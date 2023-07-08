# importacao das bibliotecas e pacotes
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# importacao dos dados referentes aos livros
books = pd.read_csv("BX-Books.csv", sep=';', encoding="latin-1", on_bad_lines="skip", low_memory=False)
# importacao dos dadsos referentes aos usuarios
users = pd.read_csv("BX-Users.csv", sep=';', encoding="latin-1", on_bad_lines="skip", low_memory=False)
# importacao ratings
ratings = pd.read_csv("BX-Book-Ratings.csv", sep=';', encoding="latin-1", on_bad_lines="skip", low_memory=False)

# rename das colunas
books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']]

books.rename(columns= {'Book-Title':'title', 'Book-Author':'author', 'Year-Of-Publication':'year', 'Publisher':'publisher'}, inplace=True)
users.rename(columns= {'User-ID':'user_id', 'Location':'location', 'Age':'age'}, inplace=True)
ratings.rename(columns= {'User-ID':'user_id', 'Book-Rating':'rating'}, inplace=True)

# qtde de ratings por usuario
ratings['user_id'].value_counts()

# usuarios que avaliaram mais de 200 livros
idUsuarios200 = ratings['user_id'].value_counts() > 200

#quantidade de usuarios que fizeram mais de 200 avaliacoes
qtdeUsuarios = idUsuarios200[idUsuarios200].index

# trazendo os ratings dos caras que avaliaram mais de 200 livros
ratings = ratings[ratings['user_id'].isin(qtdeUsuarios)]

# juntando tabelas (join ou merge)
rating_with_books = ratings.merge(books, on='ISBN')
rating_with_books.head()

#quantidade de vezes em que um livro foi avaliado
number_rating = rating_with_books.groupby('title')['rating'].count().reset_index()

# renomear campo de qtde de avaliacoes
number_rating.rename(columns= {'rating':'number_of_ratings'}, inplace=True)

# juntando todas as tabelas (livros - ratings - qtde de ratings)
final_rating = rating_with_books.merge(number_rating, on='title')

# DECISAO DE NEGOCIO

# filtrar livros que tenham pelo menos 50 avaliacoes
final_rating = final_rating[final_rating['number_of_ratings'] >= 50]

#descartar valores duplicados
final_rating.drop_duplicates(['user_id', 'title'], inplace=True)

# TRANSPOSICAO DE LINHAS - cria uma matriz livro x usuario com o valor da avaliacao
book_pivot = final_rating.pivot_table(columns='user_id', index='title', values='rating')
book_pivot.fillna(0, inplace=True)

# converter a tabela em uma matriz esparsa para que não consuma tanto do computador
book_sparse = csr_matrix(book_pivot)


# CRIACAO DA MAQUINA PRIMITIVA
model = NearestNeighbors(algorithm='brute')
model.fit(book_sparse) #treinar a maquina

#BUSCAR QUAIS SAO AS SUGESTOES DE ACORDO COM O INDEX DO LIVRO
#exemplo - livro 1984 (index 0)
distances, suggestions = model.kneighbors(book_pivot.iloc[0, :].values.reshape(1, -1))

for i in range(len(suggestions)):
    print(book_pivot.index[suggestions[i]])

#exemplo - livro HP (index 238)
#distances, suggestions = model.kneighbors(book_pivot.iloc[238, :].values.reshape(1, -1))

#for i in range(len(suggestions)):
#    print(book_pivot.index[suggestions[i]])
