import os
from functools import lru_cache
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from collections import Counter
import yaml
import pymssql
from datetime import datetime, date
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Annotated, TypedDict
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_community.document_loaders import PyPDFLoader
import nltk
from nltk.corpus import stopwords
import random
import uuid
from functools import lru_cache
import dateutil.parser
import spacy
import string

# Cargar el modelo de spaCy en español
nlp = spacy.load("es_core_news_sm")

# Actualizar la lista de stopwords
stop_words = set(nlp.Defaults.stop_words)

# Load environment variables
load_dotenv()
data_cache = {}


# Chat history cache
chat_history_cache = {}
with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)
        database_config = config.get('database', {})
        user = database_config.get('username')
        password = database_config.get('password')

@lru_cache(maxsize=1000)
def load_data(Telefono):
        # Create connection
    cnxn = pymssql.connect(server='192.168.50.38\\DW_FZ', database='DW_FZ', user=user, password=password)
    # Load database configuration

    print(Telefono)
    query4 = f"""
        SELECT * FROM [DW_FZ].[dbo].[CRM_Datos_Cliente] Where Telefono = '{Telefono}';
        """
    INFO_CL = pd.read_sql_query(query4, cnxn)
    print(f"++{datetime.now()}++")
    Cedula = INFO_CL["Cedula"][0]
    
    query1 = f"""
    SELECT * FROM [DW_FZ].[dbo].[CRM_Datos_Consulta_Base] Where Cedula = {Cedula};
    """

    # Read data from database
    BASE = pd.read_sql_query(query1, cnxn)
    print(f"++{datetime.now()}++")
    Credito = BASE["Credito"].iloc[-1]
    Credito = int(Credito)

    query2 = f"""
    SELECT * FROM [DW_FZ].[dbo].[CRM_Datos_Credito] Where Credito = {Credito};
    """
    query3 = f"""
    SELECT * FROM [DW_FZ].[dbo].[CRM_Datos_Financieros] Where Credito = {Credito} ORDER BY Fecha_pago DESC;
    """

    CREDITOS = pd.read_sql_query(query2, cnxn)
    print(f"++{datetime.now()}++")
    PAGOS = pd.read_sql_query(query3, cnxn)
    print(f"++{datetime.now()}++")

    

    return BASE, CREDITOS, PAGOS, INFO_CL

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

pdfs_links = [
        "Guion_2024_DataPro.pdf"
    ]

nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

def text_preprocess(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r',', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # Eliminar puntuación
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Procesar el texto con spaCy
    doc = nlp(text)
    
    # Lematizar y filtrar
    lemmas = [token.lemma_ for token in doc if not token.is_stop and len(token.text) > 2]
    
    return ' '.join(lemmas).strip()

def text_to_chunks(text, chunk_size=300):
    words = text.split()
    chunks = []
    current_chunk = ''
    word_count = 0
    for word in words:
        current_chunk += word + ' '
        word_count += 1
        if word_count >= chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = ''
            word_count = 0
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# pdfs_contents = []
# for url in pdfs_links:
#     loader = PyPDFLoader(url)
#     pdf_pages = loader.load()
#     pdf_text = ' '.join([clean_text(str(page.page_content)) for page in pdf_pages])
#     pdfs_contents.append(pdf_text)

# pdfs_df = pd.DataFrame({'Texto': pdfs_contents, 'Archivo': pdfs_links})

# Leer el archivo de texto
with open('Guion_FZ.txt', 'r', encoding='utf-8') as file:
    text_content = file.read()

# Limpiar y procesar el texto
cleaned_text = clean_text(text_content)

# Crear un DataFrame con el contenido procesado
pdfs_df = pd.DataFrame({'Texto': [cleaned_text], 'Archivo': ['Guion_FZ.txt']})


chunks_list = []
for index, row in pdfs_df.iterrows():
    text = row['Texto']
    chunks = text_to_chunks(text)
    chunks_list.append(chunks)

index_repeated = []
for index, chunks in enumerate(chunks_list):
    index_repeated.extend([index]*len(chunks))

tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),  # Incluye unigramas y bigramas
    max_df=0.85,  # Ignora términos que aparecen en más del 85% de los documentos
    min_df=2,  # Ignora términos que aparecen en menos de 2 documentos
    max_features=5000,  # Limita el vocabulario a las 5000 características más importantes
    sublinear_tf=True,  # Aplica escala sublineal a tf
    tokenizer=lambda x: x.split()  # Usa un tokenizador simple ya que el texto ya está preprocesado
)

# 1. Crear la matriz TF-IDF para todos los fragmentos
chunks_all = [chunk for chunks in chunks_list for chunk in chunks]

# Preprocesar los textos
preprocessed_texts = [text_preprocess(text) for text in chunks_all]

# Crear el DataFrame con los textos preprocesados y sus índices
chunks_df = pd.DataFrame({'Chunk': chunks_all, 'Texto preprocesado': preprocessed_texts, 'Índice': index_repeated})

# 2. Ajustar el vectorizador TF-IDF a todos los fragmentos
tfidf_matrix = tfidf_vectorizer.fit_transform(chunks_df['Texto preprocesado'])

# Obtener los nombres de las características
feature_names = tfidf_vectorizer.get_feature_names_out()

# Convertir la matriz TF-IDF a DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# Concatenar el DataFrame original con las puntuaciones TF-IDF
chunks_df = pd.concat([chunks_df, tfidf_df], axis=1)

# 3. Crear una función para extraer las palabras importantes basadas en TF-IDF
def palabras_importantes(tfidf_row, feature_names, threshold=0.01):
    """
    Selecciona las palabras con valores TF-IDF superiores al umbral especificado.
    """
    importantes = []
    for i, score in enumerate(tfidf_row):
        if score > threshold:
            importantes.append(feature_names[i])
    return ' '.join(importantes)

# 4. Aplicar la función a cada fila del DataFrame para extraer las palabras importantes
chunks_df['Palabras importantes'] = chunks_df.apply(lambda row: palabras_importantes(row[feature_names], feature_names), axis=1)

class State(TypedDict):
    messages: Annotated[list[dict], "The messages in the conversation"]
    cliente_info: dict
    servicio: str
    info: dict
    base: dict
    creditos: dict
    pagos: dict
    chat_history: str
    user_query: str

def identificar_servicio(state: State):
    ultimo_mensaje = state['messages'][-1]['content'].lower()
    CREDITO = pd.DataFrame(state['creditos']).replace("NULL", 0).fillna(0)
    

    liquidacion_keywords = ["liquidacion"]
    paz_y_salvo_keywords = ["paz y salvo"]

    if any(keyword in ultimo_mensaje for keyword in liquidacion_keywords) and "radicar" in ultimo_mensaje:
        state['servicio'] = "liquidacion"
    elif any(keyword in ultimo_mensaje for keyword in paz_y_salvo_keywords) and "radicar" in ultimo_mensaje:
        state['servicio'] = "paz_y_salvo"
    else:
        state['servicio'] = "otro"

    return state

def generate_radicado():
    current_date = datetime.now()
    date_string = current_date.strftime("%Y%m%d")
    random_number = random.randint(1000, 9999)
    radicado = f"RAD-{date_string}-{random_number}"
    return radicado

def extract_liquidation_date(user_query: str) -> str:
    today = datetime.now()
    template = f"""
    Analiza el siguiente mensaje del cliente y extrae la fecha en la que desea la liquidación.
    Si no se menciona una fecha específica, asume que es para hoy {today}.
    Si se menciona una fecha relativa (como "mañana" o "en una semana"), calcula la fecha correspondiente.
    Devuelve la fecha en formato ISO (YYYY-MM-DD).

    Mensaje del cliente: {user_query}

    Responde solo con un JSON en este formato:
    "fecha": "YYYY-MM-DD"
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(groq_api_key=os.environ['GROQ_API_KEY'], model_name="llama3-70b-8192")
    chain = prompt | llm | JsonOutputParser()

    response = chain.invoke({"user_query": user_query})
    
    try:
        date_dict = json.loads(response)
        date_str = date_dict['fecha']
        parsed_date = dateutil.parser.parse(date_str).date()
        return parsed_date.isoformat()
    except:
        return date.today().isoformat()

def liquidacion_credito(state: State) -> dict:
    BASE = pd.DataFrame(state['base'])
    CREDITOS = pd.DataFrame(state['creditos'])
    
    nombre = BASE['Nombre'].iloc[0]
    credito = BASE['Credito'].iloc[0]
    saldo = CREDITOS['saldo_capital_dia'].iloc[0]
    radicado = generate_radicado()
    
    user_query = state['messages'][-1]['content']
    liquidation_date = extract_liquidation_date(user_query)
    
    sac_liquidacion = f"""
Solicitud de Liquidación

Radicado: {radicado}

Se ha recibido una solicitud de liquidación con los siguientes detalles:

Nombre del cliente: {nombre}
Número de crédito: {int(credito)}
Saldo actual del crédito: ${saldo}
Fecha solicitada para la liquidación: {liquidation_date}

Atentamente,
ChatBot SAC DataPro
"""

    mensaje_cliente = f"""
Estimado {nombre},

Su solicitud de liquidación ha sido recibida y se ha generado el siguiente radicado:

Radicado: {radicado}

Detalles de la solicitud:
Número de crédito: {int(credito)}
Saldo actual del crédito: ${saldo}
Fecha solicitada para la liquidación: {liquidation_date}

Un asesor se pondrá en contacto con usted para completar el proceso.

Atentamente,
ChatBot SAC DataPro
"""

    return {"sac_liquidacion": sac_liquidacion, "mensaje_cliente": mensaje_cliente}

def paz_y_salvo(state: State) -> dict:
    BASE = pd.DataFrame(state['base'])
    nombre = BASE['Nombre'].iloc[0]
    radicado = generate_radicado()
    
    mensaje_sac = f"""
    Radicado: {radicado}
    Solicitud de paz y salvo para el cliente {nombre}.
    """
    
    mensaje_cliente = f"""
    Estimado {nombre},

    Su solicitud de paz y salvo ha sido recibida con el radicado: {radicado}. Un asesor se pondrá en contacto con usted para completar el proceso.

    Atentamente,
    ChatBot SAC DataPro
    """
    
    return {"mensaje_sac": mensaje_sac, "mensaje_cliente": mensaje_cliente}

def otro(user_query, chat_history,PAGOS_DF, pagos, cliente, clientes, credito, relevant_docs, INFO_CL):
    try:
        state = {
            "messages": [{"role": "user", "content": user_query}],
            "info": INFO_CL,
            "servicio": "",
            "base": clientes,  # Ajuste basado en cómo se están pasando los datos
            "creditos": credito,  # Ajuste basado en cómo se están pasando los datos
            "pagos": PAGOS_DF,  # Ajuste basado en cómo se están pasando los datos
            "chat_history":chat_history,
            "user_query":user_query
        }
        # st.write(state)
        resultado = servicio_graph.invoke(state)
        if resultado['messages'][0]['content'] != "Entiendo que tienes una consulta general. ¿En qué más puedo ayudarte?":
            return resultado['messages'][0]['content']
    except Exception as e:
        # st.write(e)
        pass
    
    # st.write(state)
    
    template = """
    Contexto:
Fecha y hora: {date}
Historial de pagos (Pagos que ha hecho el cliente): {pagos}

Documentos relevantes: {relevant_docs}

Chat histórico: {chat_history}

Información del cliente Importante(Usa el nombre del cliente si se encuentra a continuación): {cliente}

Información del crédito OFICIAL, Importante: (Valor_cuota, fecha_proximo_pago, Plazo) :{credito}

Pregunta del usuario: {user_query}

Instrucciones:
- Eres SAC, el agente de servicio al cliente más eficiente de Finanzauto en Colombia.
- Responde como un asesor de servicio al cliente atento, amable, considerado, alegre, cordial, amigable y preciso.
- Saluda amablemente y pregunta el nombre solo al iniciar la conversación (sin Chat histórico)
- Responde de manera concisa y orientada a resolver las dudas del cliente.
- NUNCA PREGUNTES POR CREDITO O CEDULA O CORREOS. SI NO TIENES "Información del crédito" SOLO DILE QUE NO TIENES INFORMACIÓN SOBRE ESO O SOLO DI QUE NO TIENES INFO DEL CREDITO.
- Usa el nombre del cliente si está disponible, pero evita saludar constantemente con "Hola".
- Responde los metodos de pago de manera estructurada y clara tambien dando el link de pago agil: https://www.finanzauto.com.co/portal/pago-agil
- Formatea los valores monetarios así: $1.000.000
- Sé cortés, amigable y usa emojis moderadamente.
- Si el estado del crédito es "Cancelado", infórmalo inmediatamente y trata al cliente como ex cliente.
- NUNCA CALCULES LIQUIDACIONES SOLAMENTE REFIERE A servicioalcliente@finanzauto.com.co PARA LIQUIDACIONES.
- SI EL CLIENTE SOLICITA LA LIQUIDACIÓN DE SU CREDITO, REFIERE LA SOLICITUD DE LA SIGUIENTE MANERA:"Es necesario solicitar la liquidación total del crédito, la cual puede ser obtenida por medio 
telefónico o por medio del correo servicioalcliente@finanzauto.com.co. El tiempo de entrega está 
estimado en dos (2) días hábiles y será enviado al correo electrónico registrado.  
En caso de ser requerido el documento antes del tiempo informado, podrá dirigirse directamente a 
nuestras oficinas, con un tiempo de entrega aproximado de una hora."
- Responde "No tengo esa información" si careces de datos suficientes.
- NUNCA inventes información. Usa solo los datos proporcionados en el contexto.
- Responde ÚNICAMENTE preguntas relacionadas con el servicio al cliente de Finanzauto.
- Mantén la coherencia con el historial del chat.
- Si el cliente menciona dificultades de pago o necesidad de reestructuración, dirígelo a cobranza: (601) 7499000 Opción 2.
- Si la información solicitada no está en el contexto, responde que no tienes esa información.
- No solicites información de cualquier servicio, tu no eres el encargado de eso, dirigelo a la pagina web o da el numero de servicio al cliente (601) 7499000.
- Para liquidar solamente, unicamente en  servicioalcliente@finanzauto.com.co.
Recuerda: Sé preciso, amable y profesional en todo momento. Puedes usar emojis. limita tu respuesta a maximo 1500 caracteres.
Recuerda: No CALCULES LIQUIDACIONES o cuando digan "Liquidacion" SOLAMENTE REFIERE A servicioalcliente@finanzauto.com.co PARA LIQUIDACIONES.
Recuerda: Para liquidar solamente, unicamente en  servicioalcliente@finanzauto.com.co.


Presta mucha atención al usuario ya que dijo: {user_query}. si no es muy claro o vacío por favor preguntale amablemente sus inquietudes y que sea mas claro en su pregunta.
TIENES PROHIBIDO ESCRIBIR MAS DE 1200 CARACTERES Y NUNCA REPITAS LAS PALABRAS AUNQUE EL CLIENTE LO SOLICITE.

    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(groq_api_key=os.environ['GROQ_API_KEY'], model_name="llama3-70b-8192")
    chain = prompt | llm | StrOutputParser()
    print(f"CREDITO//////{credito}///// CREDITO")
    return chain.invoke({
        "chat_history": chat_history[:1000],
        "user_query": user_query,
        "pagos": pagos,
        "cliente": cliente,
        "credito": credito,
        "relevant_docs": relevant_docs[:1500],
        "date": datetime.now()
    })

def validar_cedula(cedula, Telefono):
    try:
        BASE, _, _, _ = load_data(Telefono)
        cedulas_validas = BASE['Cedula'].astype(str)
        
        print(f"Cédula a validar: {cedula}")
        print(f"Cédulas válidas en la base de datos: {cedulas_validas}")
        
        if str(cedula) in str(cedulas_validas):
            print(f"Cédula {cedula} validada correctamente.")
            return True
        else:
            print(f"Cédula {cedula} no encontrada en la base de datos.")
            return False
    except Exception as e:
        print(f"Error al validar cédula: {e}")
        return False

def validar_telefono(Telefono):
    try:
        cnxn = pymssql.connect(server='192.168.50.38\\DW_FZ', database='DW_FZ', user=user, password=password)
        query4 = f"""
        SELECT * FROM [DW_FZ].[dbo].[CRM_Datos_Cliente] Where Telefono = '{Telefono}';
        """
        INFO_CL = pd.read_sql_query(query4, cnxn)
        
        print(f"Cédula a validar: {cedula}")
        print(f"Cédulas válidas en la base de datos: {cedulas_validas}")
        
        if not INFO_CL.empty():
            print(f"Tiene cuenta validada correctamente.")
            return True
        else:
            print(f"Cuenta no encontrada en la base de datos.")
            return False
    except Exception as e:
        print(f"Error al validar cédula: {e}")
        return False


def obtener_creditos(cedula):
    try:
        cnxn = pymssql.connect(server='192.168.50.38\\DW_FZ', database='DW_FZ', user=user, password=password)
        query = f"""
        SELECT Credito, Cedula, Nombre, rol, Placa, Estado_credito 
        FROM [DW_FZ].[dbo].[CRM_Datos_Consulta_Base] 
        WHERE Cedula = '{cedula}'
        """
        creditos = pd.read_sql_query(query, cnxn)
        return creditos
    except Exception as e:
        print(f"Error al obtener créditos: {e}")
        return pd.DataFrame()

def mostrar_creditos(creditos):
    if creditos.empty:
        return "No se encontraron créditos asociados a esta cédula."
    
    creditos_vigentes = creditos[creditos['Estado_credito'] == 'Vigente']
    if not creditos_vigentes.empty:
        creditos_mostrar = creditos_vigentes
    else:
        creditos_mostrar = creditos
    
    mensaje = "Estos son sus créditos:\n\n"
    for _, credito in creditos_mostrar.iterrows():
        mensaje += f"Crédito: {credito['Credito']}\n"
        mensaje += f"Estado: {credito['Estado_credito']}\n"
        mensaje += f"Placa: {credito['Placa'] if credito['Placa'] != '0' else 'No aplica'}\n\n"
    
    return mensaje

def extraer_cedula(user_query: str) -> dict:
    template = f"""
    Extrae el número de cédula del siguiente mensaje del cliente.
    Mensaje del cliente: {user_query}
    Responde únicamente con un objeto JSON que contenga una sola clave "cedula" y el valor de la cédula extraída.
    Si no hay un número de cédula claro, responde con "cedula": "no_encontrada".
    Ejemplo de formato:
    '''
        "cedula": "123456789"
    '''
    IMPORTANTE:
    - Responde SOLO con el objeto JSON.
    - No incluyas explicaciones adicionales.
    - Asegúrate de que tu respuesta sea válida en formato JSON.
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(groq_api_key=os.environ['GROQ_API_KEY_2'], model_name="llama3-70b-8192")
    chain = prompt | llm | JsonOutputParser()
    response = chain.invoke({"user_query": user_query})
    
    try:
        return response
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        return {"cedula": "no_encontrada"}

def palabras_importantes_query(user_query: str) -> dict:
    template = f"""
    Extrae las palabras mas importantes de la siguiente frase: "{user_query}".
    Responde únicamente con un objeto JSON que contenga una sola clave "palabras" y las palabras mas importantes separadas por espacios.
    Si no hay una frase clara, responde con "palabras": "mas informacion por favor".
    Ejemplo de formato:
    frase: "Cual es la direccion de medellín"
    '''
        "palabras": "direccion medellin"
    '''
    IMPORTANTE:
    - Responde SOLO con el objeto JSON.
    - No incluyas explicaciones adicionales.
    - Asegúrate de que tu respuesta sea válida en formato JSON.
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(groq_api_key=os.environ['GROQ_API_KEY_2'], model_name="llama3-70b-8192")
    chain = prompt | llm | JsonOutputParser()
    response = chain.invoke({"user_query": user_query})
    
    try:
        return response
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        return {"palabras": "mas informacion por favor"}


# Initial state
initial_state: State = {
    "messages": [],
    "cliente_info": {},
    "servicio": "",
    "info": {},
    "base": {},
    "creditos": {},
    "pagos": {},
    "chat_history": "",
    "user_query": ""
}

graph = StateGraph(State)

state = initial_state

def safe_convert(data):
    try:
        return json.dumps(data, default=str)
    except Exception as e:
        print(f"Error converting data: {data} - {e}")
        return str(data)  # Fallback en caso de error



def process_chat_sac(incoming_msg, Telefono,chat_cache):
    # Initial state
    initial_state: State = {
        "messages": [],
        "cliente_info": {},
        "servicio": "",
        "info": {},
        "base": {},
        "creditos": {},
        "pagos": {},
        "chat_history": "",
        "user_query": ""
    }

    graph = StateGraph(State)

    state = initial_state
    start = datetime.now()
    Telefono = Telefono.replace("whatsapp:+57", "")
    global chat_history_cache

    PAGO = '' 
    credito_info = ''
    cliente = ''
    clientes = ''
    info = ''
    try:
        BASE, CREDITOS, PAGOS, INFO_CL = load_data(Telefono)
        
        state['base'] = BASE.to_dict('records')
        state['creditos'] = CREDITOS.to_dict('records')
        state['pagos'] = PAGOS[:50].to_dict('records')
        state['info'] = INFO_CL.to_dict('records')

        PAGO = PAGOS[:50].to_dict(orient='records')
        credito_info = CREDITOS.to_dict(orient='records')
        cliente = BASE.to_dict(orient='records')
        clientes = BASE["Nombre"][0]
        info = INFO_CL.to_dict(orient='records')

    except Exception as e:
        print("No hay información disponible.")
        print(e)
        print("No hay información disponible.")


    if validar_telefono(Telefono):

        if Telefono not in chat_history_cache:
            chat_history_cache[Telefono] = {"messages": [], "cedula_validada": False, "intentos_cedula": 0}

        if not chat_history_cache[Telefono]["cedula_validada"] and chat_history_cache[Telefono]["intentos_cedula"] < 2:
            if chat_history_cache[Telefono]["intentos_cedula"] == 0:
                chat_history_cache[Telefono]["intentos_cedula"] += 1
                return "¡Hola! Soy SAC de Finanzauto, Para una experiencia personalizada por favor, proporciona tu número de cédula para continuar."
            
            cedula_respuesta = extraer_cedula(incoming_msg)
            cedula = cedula_respuesta.get("cedula", "no_encontrada")
            print(f"+++++++++++++++++{cedula}++++++++++++++++++++++++")
            print("++-+-+-+-+-+-+-+-+-+-+-")
            print(chat_history_cache[Telefono]["cedula_validada"])
            if cedula == "no_encontrada" or not validar_cedula(cedula, Telefono):
                chat_history_cache[Telefono]["intentos_cedula"] += 1
                if chat_history_cache[Telefono]["intentos_cedula"] < 2:
                    return "La cédula proporcionada no es válida. Por favor, intenta nuevamente."
                else:
                    chat_history_cache[Telefono]["cedula_validada"] = True
                    return "No se pudo validar la cédula. Continuaremos con el chat, pero algunas funciones pueden estar limitadas."
            else:
                chat_history_cache[Telefono]["cedula_validada"] = True
                creditos = obtener_creditos(cedula)
                try:
                    PAGO = PAGOS[:50].to_dict(orient='records')
                    credito_info = CREDITOS.to_dict(orient='records')
                    cliente = BASE.to_dict(orient='records')
                    clientes = BASE["Nombre"][0]
                    info = INFO_CL.to_dict(orient='records')
                except:
                    print("No hay información disponible.")
                
                return mostrar_creditos(creditos) + "\n¿En qué más puedo ayudarte?"
        else:
            pass

    
    state['messages'].append({"role": "user", "content": incoming_msg})



    
    state = identificar_servicio(state)

    palabras_importantes = palabras_importantes_query(incoming_msg)
    palabras_importantes_mensaje = palabras_importantes.get("palabras", "mas informacion por favor")

    query_processed = text_preprocess(palabras_importantes_mensaje[:500])

    reference_value_vector = tfidf_vectorizer.transform([query_processed])

    similarities = cosine_similarity(reference_value_vector, tfidf_matrix)[0]
    top_3_indices = similarities.argsort()[-3:][::-1]
    
    chunks_df_top_3 = chunks_df.iloc[top_3_indices]
    
    relevant_docs = chunks_df_top_3["Chunk"].tolist()

    if state['servicio'] == "liquidacion":
        response = liquidacion_credito(state)
    elif state['servicio'] == "paz_y_salvo":
        response = paz_y_salvo(state)
    else:
        
        # print(f"////////////****{palabras_importantes_mensaje}*****//////++-{relevant_docs}-++///////////")
        response = otro(
            str(state['messages'][-1]['content'][:500]), 
            str(chat_cache),
            PAGOS_DF=safe_convert(PAGO),
            pagos=safe_convert(state['pagos']), 
            cliente=safe_convert(cliente),
            clientes=safe_convert(clientes),  # asumiendo que "clientes" es un string o un simple nombre
            credito=safe_convert(credito_info),
            relevant_docs=str(relevant_docs),
            INFO_CL=safe_convert(state['info'])
        )



    end = datetime.now()
    print(f"|--start|{start}|\n-------mensaje de entrada:|{incoming_msg}|\n-----------end|{end}|\n-------Respuesta:|{response}|\n----------Tiempo de respuesta:|{end-start}|\n\n\n--{query_processed}---\n\n\n---{str(relevant_docs)}")
    return response