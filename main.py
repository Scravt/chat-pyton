from fastapi import FastAPI
from pydantic import BaseModel
from ConPDf import call_api_IA

app = FastAPI()

#Define un modelo para el cuerpo de la solicitud
class Message(BaseModel):
    text: str

#Define el endpoint que recibirá las solicitudes del front
@app.post("/chat/")
async def chat(message: Message):
#Invoca la IA con el mensaje del usuario

    respuesta = call_api_IA(message.text)
    return {"status": "200 OK", "respuesta": respuesta}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
