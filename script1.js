// script1.js
function runTask(task) {
    const text = document.getElementById("textInput").value;

    if (!text.trim()) {
        alert("Please enter some text.");
        return;
    }

    fetch(`http://localhost:5000/${task}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        const outputDiv = document.getElementById("output");
        outputDiv.style.display = "block";

        if (data.result) {
            document.getElementById("result").textContent =
                `${task.toUpperCase()}:\n${JSON.stringify(data.result, null, 2)}`;
        } else {
            document.getElementById("result").textContent =
                `Error: ${data.error || 'Unknown error'}`;
        }
    })
    .catch(error => {
        console.error("Error:", error);
        alert("An error occurred. Check console.");
    });
}
