import React, { useState, useRef, useEffect } from "react";
import { useForm } from "react-hook-form";
import axios from "axios";

const Layout = () => {
  const {
    register,
    handleSubmit,
    getValues,
    reset,
    formState: { errors },
  } = useForm();

  const [messages, setMessages] = useState([
    {
      role: "bot",
      text: "👋 Hey there! Paste a YouTube URL above and I'll watch it for you. Then ask me anything about the video!",
    },
  ]);
  const [uploaded, setUploaded] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [videoId, setVideoId] = useState("");
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const extractVideoId = (url) => {
    const match = url.match(
      /(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})/,
    );
    return match ? match[1] : null;
  };

  const handleUploadBtn = async () => {
    if (uploading) return;
    const url = getValues("url");
    if (!url) return;
    setUploading(true);
    try {
      const response = await axios.post("http://localhost:8080/url/upload", {
        url,
      });
      if (response.status === 200) {
        setUploaded(true);
        const id = extractVideoId(url);
        setVideoId(id || "");
        setMessages((prev) => [
          ...prev,
          {
            role: "bot",
            text: "✅ Video loaded successfully! I've watched the video. Go ahead and ask me anything about it!",
          },
        ]);
      } else {
        setUploaded(false);
      }
    } catch (error) {
      console.error("Error uploading video:", error);
      setUploaded(false);
      const detail =
        error?.response?.data?.detail ||
        "Failed to load the video. Please check the URL and try again.";
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          text: `❌ ${detail}`,
        },
      ]);
    } finally {
      setUploading(false);
    }
  };

  const OnSubmit = async (data) => {
    const userMsg = data.query;
    setMessages((prev) => [...prev, { role: "user", text: userMsg }]);
    reset({ url: getValues("url"), query: "" });
    setLoading(true);
    try {
      const response = await axios.post("http://localhost:8080/api/chat", {
        query: userMsg,
      });
      setMessages((prev) => [...prev, { role: "bot", text: response.data }]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: "⚠️ Something went wrong. Please try again." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center p-4 font-sans">
      {/* Ambient glow blobs */}
      <div className="fixed top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
        <div className="absolute top-[-10%] left-[-10%] w-96 h-96 bg-red-600 opacity-20 rounded-full blur-3xl" />
        <div className="absolute bottom-[-10%] right-[-5%] w-80 h-80 bg-purple-700 opacity-20 rounded-full blur-3xl" />
        <div className="absolute top-[40%] left-[50%] w-64 h-64 bg-pink-600 opacity-10 rounded-full blur-3xl" />
      </div>

      <div className="relative z-10 w-full max-w-2xl flex flex-col gap-4">
        {/* Header */}
        <div className="text-center">
          <div className="flex items-center justify-center gap-3 mb-1">
            <div className="w-10 h-10 bg-red-600 rounded-xl flex items-center justify-center shadow-lg shadow-red-600/40">
              <svg
                className="w-6 h-6 text-white"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z" />
              </svg>
            </div>
            <h1 className="text-3xl font-extrabold text-white tracking-tight">
              YouTube<span className="text-red-500">AI</span>
            </h1>
          </div>
          <p className="text-gray-400 text-sm">
            Ask anything about any YouTube video
          </p>
        </div>

        {/* Main Card */}
        <div className="bg-gray-900 border border-gray-800 rounded-2xl shadow-2xl overflow-hidden">
          {/* URL Upload Bar */}
          <div className="p-4 bg-gray-900 border-b border-gray-800">
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-widest mb-2">
              YouTube URL
            </p>
            <div className="flex gap-2">
              <div className="flex-1 relative">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500">
                  <svg
                    className="w-4 h-4"
                    fill="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z" />
                  </svg>
                </span>
                <input
                  type="text"
                  placeholder="https://www.youtube.com/watch?v=..."
                  {...register("url", {
                    required: "No URL provided",
                    pattern: {
                      value:
                        /^(https?\:\/\/)?(www\.youtube\.com|youtu\.?be)\/.+$/,
                      message: "Please enter a valid YouTube URL",
                    },
                  })}
                  className="w-full bg-gray-800 text-white text-sm placeholder-gray-500 border border-gray-700 rounded-xl pl-9 pr-4 py-2.5 focus:outline-none focus:border-red-500 focus:ring-1 focus:ring-red-500 transition-all"
                />
              </div>
              <button
                type="button"
                onClick={handleUploadBtn}
                disabled={uploading || uploaded}
                className={`px-5 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 whitespace-nowrap ${
                  uploaded
                    ? "bg-green-600 text-white cursor-default shadow-lg shadow-green-600/30"
                    : uploading
                      ? "bg-gray-700 text-gray-400 cursor-wait"
                      : "bg-red-600 hover:bg-red-500 text-white shadow-lg shadow-red-600/30 hover:shadow-red-500/40 active:scale-95"
                }`}
              >
                {uploaded
                  ? "✓ Loaded"
                  : uploading
                    ? "Loading..."
                    : "Load Video"}
              </button>
            </div>
            {errors.url && (
              <p className="text-red-400 text-xs mt-1.5">
                {errors.url.message}
              </p>
            )}

            {/* Video preview thumbnail */}
            {videoId && (
              <div className="mt-3 flex items-center gap-3 bg-gray-800 rounded-xl p-2 border border-gray-700">
                <img
                  src={`https://img.youtube.com/vi/${videoId}/mqdefault.jpg`}
                  alt="Video thumbnail"
                  className="w-20 h-14 object-cover rounded-lg"
                />
                <div>
                  <p className="text-green-400 text-xs font-semibold">
                    Video Ready
                  </p>
                  <p className="text-gray-400 text-xs mt-0.5">
                    Video loaded • Ask away!
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Chat messages */}
          <div className="h-80 overflow-y-auto p-4 flex flex-col gap-3 scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-transparent">
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex items-end gap-2 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}
              >
                {/* Avatar */}
                <div
                  className={`w-7 h-7 rounded-full flex-shrink-0 flex items-center justify-center text-xs font-bold ${
                    msg.role === "user"
                      ? "bg-gradient-to-br from-purple-500 to-pink-500 text-white"
                      : "bg-red-600 text-white"
                  }`}
                >
                  {msg.role === "user" ? "U" : "AI"}
                </div>

                {/* Bubble */}
                <div
                  className={`max-w-[78%] px-4 py-2.5 rounded-2xl text-sm leading-relaxed ${
                    msg.role === "user"
                      ? "bg-gradient-to-br from-purple-600 to-pink-600 text-white rounded-br-sm shadow-lg shadow-purple-600/20"
                      : "bg-gray-800 text-gray-100 border border-gray-700 rounded-bl-sm"
                  }`}
                >
                  {msg.text}
                </div>
              </div>
            ))}

            {/* Typing indicator */}
            {loading && (
              <div className="flex items-end gap-2">
                <div className="w-7 h-7 rounded-full bg-red-600 flex-shrink-0 flex items-center justify-center text-xs font-bold text-white">
                  AI
                </div>
                <div className="bg-gray-800 border border-gray-700 px-4 py-3 rounded-2xl rounded-bl-sm flex gap-1 items-center">
                  <span
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0ms" }}
                  />
                  <span
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: "150ms" }}
                  />
                  <span
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: "300ms" }}
                  />
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input bar */}
          <form
            onSubmit={handleSubmit(OnSubmit)}
            className="p-4 border-t border-gray-800 bg-gray-900"
          >
            {errors.query && (
              <p className="text-red-400 text-xs mb-2">
                {errors.query.message}
              </p>
            )}
            <div className="flex gap-2 items-center">
              <input
                type="text"
                placeholder={
                  uploaded
                    ? "Ask anything about the video..."
                    : "Load a video first to start chatting..."
                }
                {...register("query", { required: "Enter a message first!" })}
                disabled={!uploaded}
                className="flex-1 bg-gray-800 text-white text-sm placeholder-gray-500 border border-gray-700 rounded-xl px-4 py-3 focus:outline-none focus:border-red-500 focus:ring-1 focus:ring-red-500 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
              />
              <button
                type="submit"
                disabled={!uploaded || loading}
                className="w-11 h-11 rounded-xl bg-red-600 hover:bg-red-500 disabled:bg-gray-700 disabled:cursor-not-allowed flex items-center justify-center transition-all duration-200 shadow-lg shadow-red-600/30 hover:shadow-red-500/40 active:scale-95 flex-shrink-0"
              >
                <svg
                  className="w-5 h-5 text-white rotate-90"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                  />
                </svg>
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Layout;
