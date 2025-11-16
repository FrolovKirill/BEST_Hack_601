import streamlit as st
import requests
from openai import OpenAI
import pandas as pd

df = pd.read_csv('front/df_main_with_names.csv')
# ======= UI Setup =======
st.set_page_config(page_title="Address Normalizer", layout="wide")
st.title("🏠 Нормализация адресов")

# --- Сайдбар ---
st.sidebar.header("⚙️ Настройки")

# Адрес бэка
backend_url = st.sidebar.text_input(
    "Backend URL",
    value="http://localhost:8000/normalize",
    help="Эндпоинт нормализации адресов"
)

# Настройки LLM (Ollama)
api_key = st.sidebar.text_input(
    "LLM Server (base_url)",
    value="http://127.0.0.1:11434/v1",
    help="URL ollama, например http://127.0.0.1:11434/v1"
)

model_name = st.sidebar.text_input(
    "Model",
    value="gemma3:12b-it-qat"
)


# ======= Основной UI =======
address = st.text_input("Введите адрес", placeholder="Например: Тверская 10")

# Переменная для хранения нормализованного адреса
if "normalized_address" not in st.session_state:
    st.session_state["normalized_address"] = None
if "llm_answer" not in st.session_state:
    st.session_state["llm_answer"] = None


# --- Кнопка Обработать ---
if st.button("Обработать"):
    if not address.strip():
        st.error("Введите адрес!")
    else:
        try:
            response = requests.post(
                backend_url,
                json={"address": address},
                timeout=10
            )
            response.raise_for_status()

            st.session_state["normalized_address"] = response.json().get("normalized_address")
            address_block = list(df[df['id'] == response.json().get("normalized_address")['id']]['name'])[0]
            st.session_state["address_block"] = address_block

            st.session_state["llm_answer"] = None  # сбрасываем старый ответ

        except Exception as e:
            st.error(f"Ошибка при запросе к бэкенду: {e}")


# --- Вывод нормализованного адреса ---
if st.session_state["normalized_address"]:
    st.success("Нормализованный адрес:")
    st.code(st.session_state["normalized_address"], language="text")

    st.write("")  # небольшой отступ

    # --- Кнопка "Детальная информация" ---
    if st.button("Детальная информация"):
        if not api_key:
            st.error("Введите API key в боковой колонке!")
        else:
            try:
                client = OpenAI(base_url=api_key, api_key='ollama')

                with st.spinner("Обращение к LLM..."):
                    prompt = (
                        f'''Вот адрес и информация об этом адресе, полученная из OpenStreetMap.

Твоя задача:
- Кратко описать объект: что это за дом, какие организации находятся внутри, любая другая фактическая информация.
- Не добавляй ничего от себя. Используй только те данные, которые даны во входе.
- Если кроме адреса и координат в данных ничего нет, прямо скажи, что дополнительной информации нет.
- Ответ должен быть коротким: 2–4 предложения.

Данные об адресе:
{st.session_state["address_block"]}'''
                    )
                    print(prompt)

                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=200,
                        temperature=0.
                    )

                    answer = completion.choices[0].message.content
                    st.session_state["llm_answer"] = answer

            except Exception as e:
                st.error(f"Ошибка при обращении к LLM: {e}")


# --- Вывод результата LLM ---
if st.session_state["llm_answer"]:
    st.markdown("### 💬 Детальная информация")
    st.write(st.session_state["llm_answer"])
