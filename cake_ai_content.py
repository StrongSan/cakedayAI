import pandas as pd
import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

#케이크 추천 알고리즘 - 상위5개만 일단..
def recommend_cakes(cake_id, top_n = 5):
    cake_idx = cake_id - 1
    similarity_scores = list(enumerate(similarity_matrix[cake_idx]))

    #enumerate 하면 (인덱스,값 이렇게 나오니까 lambda로 값만 채용)
    similarity_scores = sorted(similarity_scores, key = lambda x: x[1], reverse = True)
    top_matches = similarity_scores[:top_n]

    recommended = df.iloc[[i[0] for i in top_matches]]
    recommended["유사도"] = [round(score,3) for _, score in top_matches]
    return recommended


#Mysql에 직접 연결
conn = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "Rlarkdtks94302.",
    database = "cakeday"
)

#데이터 확인 - 총 181개 나오고, 고유 태그는 16개 나와야함
query = "select * from cakes"
df = pd.read_sql(query, conn)

df['tags'] = df[['size', 'filling', 'sheet', 'type']].agg(' '.join, axis = 1)

#태그만 벡터화
vectorizer = TfidfVectorizer()
cake_vectors = vectorizer.fit_transform(df['tags'])

similarity_matrix = cosine_similarity(cake_vectors)




print('-' * 100)
print("벡터 행렬 크기: ", cake_vectors.shape)
print('-' * 100)
print("고유 태그 목록: \n", vectorizer.get_feature_names_out())
print('-' * 100)
print("유사도 행렬 크기: ",similarity_matrix.shape)
print('-' * 100)
n = int(input("찾고싶은 케이크 번호를 입력하세요 : "))
recommended = recommend_cakes(n)
print(f"{n}번 케이크와 유사한 [전체]케이크(자신 포함) : \n", similarity_matrix[n-1])
print('-' * 100)
print(f"{n}번 케이크와 비슷한 케이크 추천: \n", 
      tabulate(recommended, headers='keys', tablefmt='pretty', showindex=False))


