# Instalación de paquetes necesarios
# Ejecuta esto en una celda separada si estás usando Jupyter Notebook

# Importaciones
import os
import json
from getpass import getpass
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Annotated
from typing_extensions import TypedDict

import re
import numpy as np
import pandas as pd
import pymssql
import openai
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import yaml
from functools import lru_cache
import random
import uuid

from pydantic import BaseModel, Field

from langgraph.graph.message import AnyMessage, add_messages
from langgraph.graph import END, StateGraph, START
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from langchain import PromptTemplate, RunnableSequence
from langchain.schema import BaseMessage

from IPython.display import Image, display

# Cargar variables de entorno
load_dotenv()

# Inicializar el recuperador de Milvus
retriever = MilvusRetriever(documents=[], k=3)
retriever.init()

# Cargar configuración de la base de datos
with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)
    database_config = config.get('database', {})
    user = database_config.get('username')
    password = database_config.get('password')

# Definir constantes para los nodos del grafo
LOGUER = 'loguer'
PROCESSOR = 'processor'
VERIFIER = 'verifier'
ASSISTANT = 'assistant'
HIL = 'hil'

# Definir modelos de datos usando Pydantic
class RequiredInformation(BaseModel):
    provided_id: Optional[int] = Field(None, description="La cédula que proporcionó el usuario")
    provided_email: Optional[str] = Field(None, description="El email que proporcionó el usuario")

class State(TypedDict):
    user_question: str
    messages: Annotated[List[AnyMessage], add_messages]
    validated: bool
    required_information: RequiredInformation
    telefono: int

# Definir la clase LLM personalizada para Groq
class ChatGroqWrapper(LLM):
    groq_api_key: str
    model_name: str

    class Config:
        model_config = {"arbitrary_types_allowed": True}

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Realiza una llamada a la API de Groq para obtener una respuesta basada en el prompt.
        """
        # URL de la API de Groq (reemplaza esto con la URL real proporcionada por Groq)
        api_url = f"https://api.groq.com/v1/engines/{self.model_name}/completions"

        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "prompt": prompt,
            "max_tokens": 150,
            "temperature": 0.7,
            "stop": stop
        }

        try:
            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            # Asumiendo que la respuesta sigue el formato de OpenAI
            return result.get("choices", [{}])[0].get("text", "").strip()
        except requests.exceptions.RequestException as e:
            print(f"Error en la API de Groq: {e}")
            return "Lo siento, ocurrió un error al procesar tu solicitud."

    async def _arun(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Método asincrónico para realizar la llamada a la API de Groq.
        """
        return self._call(prompt, stop)

    @property
    def _identifying_params(self) -> Dict[str, str]:
        return {
            "groq_api_key": self.groq_api_key,
            "model_name": self.model_name
        }

    @property
    def _llm_type(self) -> str:
        return "ChatGroqWrapper"

# Crear una instancia del LLM usando ChatGroqWrapper
llm = ChatGroqWrapper(
    groq_api_key=os.environ['GROQ_API_KEY'],
    model_name="llama3-70b-8192"
)

# Definir las clases de datos para el usuario y el estado
class UserData(BaseModel):
    user_id: str
    appointment_details_step: int = 0
    conversation_active: bool = True
    credit_number: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone_number: Optional[str] = None
    selected_slot: Optional[str] = None
    reason: Optional[str] = None
    available_slots: List[str] = []
    messages: List[AnyMessage] = Field(default_factory=list)

class AppState(BaseModel):
    user_data: UserData
    messages: List[AnyMessage] = Field(default_factory=list)
    current_node: Optional[str] = None

    def add_message(self, message: AnyMessage):
        self.messages.append(message)
        self.user_data.messages.append(message)

# Inicializar la base de datos
def init_db():
    conn = pymssql.connect(server='192.168.50.38\\DW_FZ', database='DW_FZ', user=user, password=password)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS citas (
            id INT PRIMARY KEY IDENTITY(1,1),
            user_id NVARCHAR(50),
            credit_number NVARCHAR(50),
            first_name NVARCHAR(50),
            last_name NVARCHAR(50),
            phone_number NVARCHAR(50),
            reason NVARCHAR(255),
            appointment_datetime NVARCHAR(50)
        )
    ''')
    conn.commit()
    conn.close()

# Función para guardar la cita
def save_appointment(user_data: UserData):
    conn = pymssql.connect(server='192.168.50.38\\DW_FZ', database='DW_FZ', user=user, password=password)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO citas (
            user_id, credit_number, first_name, last_name,
            phone_number, reason, appointment_datetime
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    ''', (
        user_data.user_id,
        user_data.credit_number,
        user_data.first_name,
        user_data.last_name,
        user_data.phone_number,
        user_data.reason,
        user_data.selected_slot
    ))
    conn.commit()
    conn.close()

# Función para detectar intención de agendar cita
def detect_intent_schedule_appointment(user_input: str) -> bool:
    triggers = ["agendar una cita", "quiero hablar con una persona", "programar cita", "hablar con un asesor"]
    return any(trigger in user_input.lower() for trigger in triggers)

# Definir el prompt del asistente
system_prompt = """Eres un asistente auxiliar con la tarea de verificar la identidad del cliente.
1. Primero necesitas recoger la información del cliente para poder verificarlo.
2. Luego de que colectes toda la información, di amablemente gracias y que vas a pasar a verificarlo.

La información a continuación es la que debes recolectar:

class RequiredInformation(BaseModel):
    provided_id: Optional[int] = Field(description="La cédula que proporcionó el usuario")
    provided_email: Optional[str] = Field(description="El email que proporcionó el usuario")
    
Asegurate de tener la información antes de que puedas proceder, pero recolectala un campo a la vez. 
Si el usuario se equivocó ingresando los datos, por favor dile porqué y que vuelva a ingresar el dato.
Si alguna de esta información no es proporcionada retorna None

NO LLENES LA INFORMACIÓN DEL USUARIO, RECOLECTALA."""

assistant_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        (
            "human",
            "User question: {user_question}\n"
            "Chat history: {messages}\n"
            "Telefono: {telefono}\n\n"
            "What the user has provided so far: {provided_required_information}\n\n",
        ),
    ]
)

# Definir la cadena de obtención de información
def assistant_node(state: AppState) -> Dict:
    print("En el logueador", state.dict())
    get_information_chain = RunnableSequence([
        assistant_prompt,
        llm
    ])
    res = get_information_chain.run(
        {
            "user_question": state.user_data.user_id,  # Ajustar según sea necesario
            "messages": state.messages,
            "telefono": state.user_data.telefono,
            "provided_required_information": state.user_data.required_information.dict() if state.user_data.required_information else {}
        }
    )
    # Asumimos que la respuesta del LLM es el siguiente mensaje del asistente
    return {
        "user_question": state.messages[-1].content if state.messages else "",
        "messages": [AIMessage(content=res)]
    }

# Función para combinar información requerida
def combine_required_info(info_list: List[RequiredInformation]) -> RequiredInformation:
    print("Combinando información requerida...")
    info_list = [info for info in info_list if info is not None]

    if len(info_list) == 1:
        return info_list[0]
    combined_info = {}
    for info in info_list:
        for key, value in info.dict().items():
            if value is not None:
                combined_info[key] = value
    print(combined_info)
    return RequiredInformation(**combined_info)

# Función para procesar información ingresada por el usuario
def process_info(state: AppState) -> Dict:
    print("Procesando información ingresada por el usuario:", state.dict())
    structured_llm_user_info = RunnableSequence([
        assistant_prompt,
        llm  # En lugar de with_structured_output
    ])
    
    res = structured_llm_user_info.run(
        {
            "user_question": state.user_data.user_id,
            "messages": state.messages,
            "telefono": state.user_data.telefono,
            "provided_required_information": state.user_data.required_information.dict() if state.user_data.required_information else {}
        }
    )
    # Asumimos que la respuesta del LLM es un RequiredInformation
    new_info = RequiredInformation(provided_id=res.get("provided_id"), provided_email=res.get("provided_email"))
    if "required_information" in state.user_data:
        required_info = combine_required_info(
            info_list=[new_info, state.user_data.required_information]
        )
    else:
        RequiredInformation(**new_info)            
    return {
        "required_information": required_info,
        "messages": [HumanMessage(content=state.user_data.user_question)],
    }

# Función para verificar la información en la base de datos
def verify_information(state: AppState) -> Dict:
    print("Verificando...")
    Telefono = state.user_data.telefono
    required_information: RequiredInformation = state.user_data.required_information

    cnxn = pymssql.connect(server='192.168.50.38\\DW_FZ', database='DW_FZ', user=user, password=password)
    query4 = f"""
        SELECT * FROM [DW_FZ].[dbo].[CRM_Datos_Cliente] WHERE Telefono = '{Telefono}';
    """
    df_cl = pd.read_sql_query(query4, cnxn)
    cnxn.close()

    if not df_cl.empty:
        correo_cl = df_cl['Correo'].iloc[0]
        cedula_cl = df_cl['Cedula'].iloc[0]
        if required_information.provided_id == cedula_cl and required_information.provided_email == correo_cl:
            print("Verificado!!!")
            return {"validated": True}
        else:
            return {"validated": False}          
    else: 
        return {"validated": False}

# Función para verificar si se han proporcionado todos los detalles requeridos
def provided_all_details(state: AppState) -> str:
    print("Mirando si ya ingresó toda la info: Arista de processor-loguer")
    if not state.user_data.required_information.provided_id or not state.user_data.required_information.provided_email:
        return "need to collect more information"
    print("Ya ingresó toda la info")
    return "all information collected"

# Función para verificar la información
def verified(state: AppState) -> str:
    print("En la arista de verificación")
    verified_successfully = state.validated

    if verified_successfully:
        return "agent_with_tools"
    else:
        return ASSISTANT

# Función para preguntar al humano (placeholder)
def ask_human(state: AppState):
    # Implementa según tus necesidades
    pass

# Definir herramientas (Tools)
@tool
def lookup_questions(query: str) -> str:
    """
    Consulta la base de datos de documentos para resolver la pregunta del cliente
    """
    docs = retriever.invoke(query)
    return docs

# Definir el asistente usando RunnableSequence
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres SAC, el agente de servicio al cliente más eficiente de Finanzauto en Colombia."
            "Usa las herramientas otorgadas para responder a las preguntas del usuario, mostrar créditos, etc."
            "\n\nteléfono del usuario actual:\n<User>{telefono}</User>"
            "\nTiempo actual: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

part_1_tools = [
    lookup_questions
]

part_1_assistant_runnable = RunnableSequence([
    primary_assistant_prompt,
    llm.bind_tools(part_1_tools)  # bind_tools es hipotético, ajusta según tu implementación
])

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: AppState):
        print("En el asistente!!!")
        while True:
            result = self.runnable.run(
                {
                    "user_question": state.messages[-1].content if state.messages else "",
                    "messages": state.messages if state.messages else [],
                    "telefono": state.user_data.telefono
                }
            )
            # Manejar respuestas vacías o inválidas
            if not result.tool_calls and (not result.content or (isinstance(result.content, list) and not result.content[0].get("text"))):
                messages = state.messages + [HumanMessage(content="Responde con un output real")]
                state = AppState(
                    user_data=state.user_data,
                    messages=messages,
                    validated=state.validated,
                    required_information=state.required_information,
                    telefono=state.telefono
                )
            else:
                break
        return {"messages": result}

# Definir el flujo del grafo
workflow = StateGraph(State)
workflow.add_node(LOGUER, assistant_node)
workflow.add_node(HIL, ask_human)
workflow.add_node(PROCESSOR, process_info)
workflow.add_node(VERIFIER, verify_information)
workflow.add_node(ASSISTANT, Assistant(part_1_assistant_runnable))

workflow.set_entry_point(LOGUER)
workflow.add_edge(LOGUER, HIL)
workflow.add_edge(HIL, PROCESSOR)
workflow.add_conditional_edges(
    PROCESSOR,
    provided_all_details,
    {
        "need to collect more information": LOGUER,
        "all information collected": VERIFIER,
    },
)
workflow.add_conditional_edges(
    VERIFIER,
    verified,
    {"agent_with_tools": ASSISTANT, LOGUER: LOGUER},
)
workflow.add_edge(ASSISTANT, END)

# Compilar el grafo
memory = MemorySaver()
compiled_graph = workflow.compile(checkpointer=memory, interrupt_before=[HIL,])

# Visualizar el grafo utilizando Mermaid
compiled_graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

# Mostrar el grafo (si estás en Jupyter Notebook)
try:
    display(Image("graph.png"))
except ImportError:
    print("No se puede mostrar la imagen del grafo en este entorno.")

# Función para ejecutar el grafo
def run_graph(user_id: str):
    init_db()
    user_data = UserData(user_id=user_id, telefono=3152332041)
    state = AppState(
        user_data=user_data,
        messages=[],
        validated=False,
        required_information=RequiredInformation(),
        telefono=3152332041
    )
    config = {"configurable": {"thread_id": user_id}}

    # Iniciar la conversación agregando el mensaje inicial
    state.add_message(HumanMessage(content="Hola, soy daniel"))

    while state.user_data.conversation_active:
        try:
            outputs = compiled_graph.invoke(input=state, config=config)
            state = outputs.state  # Actualizar el estado
        except Exception as e:
            print(f"Error al ejecutar el grafo: {e}")
            break

        # Mostrar el último mensaje del asistente
        if state.messages:
            last_message = state.messages[-1]
            if isinstance(last_message, AIMessage):
                print(f"Asistente: {last_message.content}")

        if not state.user_data.conversation_active:
            break

        # Solicitar entrada del usuario
        user_input = input("Usuario: ")
        if not user_input.strip():
            continue
        # Agregar el mensaje del usuario al estado
        state.add_message(HumanMessage(content=user_input))

# Ejecutar el grafo
if __name__ == "__main__":
    user_id = "usuario_123"
    run_graph(user_id)
