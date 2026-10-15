import { useState, useEffect, useRef } from "react";
import "./App.css";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function App() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const messagesEndRef = useRef(null);

  // Scroll to the newest message or loading indicator.
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  }, [messages, loading]);

  async function sendMessage(e) {
    e.preventDefault();

    const userMessage = message.trim();

    if (!userMessage || loading) return;

    const history = messages.map(({ role, text }) => ({
      role,
      content: text,
    }));

    setMessages((prev) => [
      ...prev,
      { role: "user", text: userMessage },
    ]);

    setMessage("");
    setLoading(true);

    try {
        const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: userMessage,
          history,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || `Request failed: ${response.status}`
        );
      }

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.response ?? "No response was returned.",
        },
      ]);
    } catch (error) {
      console.error("Chatbot request failed:", error);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: "The chatbot encountered a connection or processing error. Please try again.",
        },
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
        <div className="messages" aria-live="polite">
          {messages.length === 0 && (
            <p className="welcome">
              Hello! How can I help you learn about DevVibe?
            </p>
          )}

          {messages.map((item, index) => (
            <div
              key={index}
              className={`message ${item.role}`}
            >
              {item.role === "assistant" ? (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {item.text}
                </ReactMarkdown>
              ) : (
                item.text
              )}
            </div>
          ))}

          {loading && (
            <div
              className="message assistant loading-message"
              role="status"
              aria-label="DevVibe AI is generating a response"
            >
              <span>Thinking</span>
              <span className="loading-dots" aria-hidden="true">
                <span />
                <span />
                <span />
              </span>
            </div>
          )}

          <div ref={messagesEndRef} aria-hidden="true" />
        </div>

        <form onSubmit={sendMessage}>
          <input
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Ask about DevVibe..."
            aria-label="Your message"
            disabled={loading}
          />

          <button
            type="submit"
            disabled={loading || !message.trim()}
          >
            {loading ? "Thinking..." : "Send"}
          </button>
        </form>
      </main>
    </div>
  );
}

export default App;