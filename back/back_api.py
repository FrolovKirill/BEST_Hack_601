from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import re
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rapidfuzz import fuzz # Импортируем rapidfuzz

app = FastAPI()

# ---- Модель входных данных ----
class AddressRequest(BaseModel):
    address: str

# --- Блок загрузки (остается без изменений) ---
print("Загрузка сервиса...")
with open('back/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
tfidf_matrix = load_npz('back/tfidf_matrix.npz')
df = pd.read_pickle('back/address_data_ngram.pkl')
print("Сервис готов к работе.")

def normalize_address(address):
    """
    Приводит адрес к стандартному, "чистому" виду:
    - Нижний регистр, обработка буквы "ё".
    - Удаление знаков препинания.
    - Стандартизация сокращений (ул, г, д и т.д.).
    - Удаление всех пробелов для слитного написания.
    """
    # 1. Приводим к нижнему регистру и обрабатываем "ё"
    address = address.lower().replace('ё', 'е')
    
    # 2. Удаляем все символы, кроме букв, цифр и пробелов
    address = re.sub(r'[^а-яa-z0-9\s/]', ' ', address)
    
    # 3. Стандартизация ключевых сокращений
    replacements = {
        r'\bг\b': 'город', r'\bул\b': 'улица', r'\bд\b': 'дом',
        r'\bпр\b': 'проспект', r'\bпр-т\b': 'проспект', r'\bк\b': 'корпус',
        r'\bкорп\b': 'корпус', r'\bстр\b': 'строение', r'\bлит\b': 'литера'
    }
    for old, new in replacements.items():
        address = re.sub(old, new, address)
        
    # 4. Схлопываем множественные пробелы в один (промежуточный шаг)
    address = ' '.join(address.split())
    
    # # 5. НОВЫЙ ШАГ: Удаляем ВСЕ пробелы, чтобы получить слитное написание
    # address = re.sub(r'\s+', '', address)
    
    return address

@app.post("/normalize")
def find_coordinates_sparse_hybrid(req: AddressRequest, k=10, score_threshold=50): # ИЗМЕНЕНИЕ 1: более адекватный порог
    """
    Выполняет гибридный поиск:
    1. Находит K кандидатов по TF-IDF (быстро, на разреженной матрице).
    2. Ранжирует их с помощью fuzz.token_sort_ratio (точно, нечувствительно к порядку слов).
    """
    # --- ЭТАП 1: Быстрый поиск кандидатов ---
    query = req.address
    normalized_query = normalize_address(query)
    query_vector = vectorizer.transform([normalized_query])

    scores = (tfidf_matrix * query_vector.T).toarray()
    # ИЗМЕНЕНИЕ 2: "Расплющиваем" scores в 1D массив перед сортировкой
    scores_1d = scores.flatten()
    
    # Сортируем и берем K лучших индексов
    candidate_indices = np.argsort(scores_1d)[-k:]
    candidate_indices = np.flip(candidate_indices) # Разворачиваем, чтобы лучший был первым

    # --- ЭТАП 2: Точное ранжирование кандидатов ---
    best_match_idx = -1 # Инициализируем как -1 на случай, если ничего не найдется
    highest_score = 0

    for idx in candidate_indices:
        # Пропускаем кандидатов с нулевой схожестью на первом этапе
        if scores_1d[idx] == 0:
            continue
            
        candidate_address = df.loc[idx, 'normalized']
        
        # Вычисляем точную схожесть, нечувствительную к порядку слов
        score = fuzz.token_sort_ratio(normalized_query, candidate_address)
        
        if score > highest_score:
            highest_score = score
            best_match_idx = idx

    # --- ЭТАП 3: Проверка порога и возврат результата ---
    if highest_score >= score_threshold:
        result_row = df.loc[best_match_idx]
        normalized =  {
            # ИЗМЕНЕНИЕ 3: Используйте правильные имена колонок из вашего DataFrame!
            'found_address': result_row['address'], 
            'street': result_row['street'],
            'housenumber': int(result_row['housenumber']),
            'latitude': result_row['latitude'],
            'longitude': result_row['longitude'],
            # ИЗМЕНЕНИЕ 4: Правильная метка для score
            'score': f"{highest_score} (fuzz.token_sort_ratio)",
            'id': int(result_row['id'])
        }
    else:
        normalized = None

    return {"normalized_address": normalized}
