const handleKeyPress = (event) => {
    if (event.key === "Enter") {
        event.preventDefault(); // Prevent form submission
        sendMessage();
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const sendBtn = document.getElementById("send-btn");
    sendBtn.addEventListener("click", sendMessage);
});

window.addEventListener("keypress", (e) => handleKeyPress(e));

function sendMessage() {
    const userInput = document.getElementById("user-input").value;
    displayMessage(userInput, patient = true);

    fetch("http://localhost:5000/process", {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded",
        },
        body: `text=${encodeURIComponent(userInput)}`,
    })
        .then((response) => response.text())
        .then((data) => displayMessage(data));
}

function displayMessage(message, patient = false) {
    const chatMessages = document.getElementById("chat-messages");
    const messageElement = document.createElement("div");
    messageElement.className = patient ? "message-patient" : "message";
    messageElement.textContent = patient ? "patient: " + message : "doctor: " + message;
    chatMessages.appendChild(messageElement);
    document.getElementById("user-input").value = "";
}
