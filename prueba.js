// Crear el objeto de datos que quieres enviar
var message = {
    text: "Hola, cómo estás?"
};

// Realizar la solicitud POST
fetch('http://127.0.0.1:8000/chat/', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(message),
})
.then(response => {
    if (!response.ok) {
        throw new Error('Network response was not ok');
    }
    return response.json();
})
.then(data => {
    console.log('Estado:', data.status);
    console.log('Respuesta:', data.respuesta);
})
.catch(error => {
    console.error('Hubo un problema con la solicitud:', error);
});








