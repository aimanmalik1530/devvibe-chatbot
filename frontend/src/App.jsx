import { useState } from "react";
import "./App.css";

function App() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  async function sendMessage(e) {
    e.preventDefault();

    if (!message.trim() || loading) return;
    setLoading(true);

    const userMessage = message;
    setMessages((prev) => [...prev, { role: "user", text: userMessage }]);
    setMessage("");

    try {
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage }),
      });

      const data = await response.json();

      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: data.response },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "Unable to connect. Please try again." },
      ]);
    } finally {
      setLoading(false);
    }
  } 

  return (
    <div className="app">
      <header>
        <h1>DevVibe AI</h1>
        <p>Your assistant for DevVibe services and solutions</p>
      </header>

      <main className="chat">
        <div className="messages">
          {messages.length === 0 && (
            <p className="welcome">
              Hello! How can I help you learn about DevVibe?
            </p>
          )}

          {messages.map((item, index) => (
            <div key={index} className={`message ${item.role}`}>
              {item.text}
            </div>
          ))}
        </div>

        <form onSubmit={sendMessage}>
          <input
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Ask about DevVibe..."
            aria-label="Your message"
          />
          <button type="submit" disabled={loading}>
            {loading ? "Thinking..." : "Send"}
          </button>
        </form>
      </main>
    </div>
  );
}

export default App;