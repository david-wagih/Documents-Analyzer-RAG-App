import React from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import HRExtractor from "./pages/HRExtractor";
import DocChat from "./pages/DocChat";

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <nav className="bg-white shadow-lg">
          <div className="max-w-7xl mx-auto px-4">
            <div className="flex justify-between h-16">
              <div className="flex space-x-8">
                <Link
                  to="/"
                  className="inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium"
                >
                  Home
                </Link>
                <Link
                  to="/hr-extractor"
                  className="inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium"
                >
                  HR Letter Extractor
                </Link>
                <Link
                  to="/doc-chat"
                  className="inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium"
                >
                  Documentation Chat
                </Link>
              </div>
            </div>
          </div>
        </nav>

        <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          <Routes>
            <Route path="/hr-extractor" element={<HRExtractor />} />
            <Route path="/doc-chat" element={<DocChat />} />
            <Route path="/" element={<Home />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

function Home() {
  return (
    <div className="text-center">
      <h1 className="text-4xl font-bold mb-8">Welcome to Document Analyzer</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <Link
          to="/hr-extractor"
          className="p-6 border rounded-lg shadow-lg hover:shadow-xl transition-shadow"
        >
          <h2 className="text-2xl font-semibold mb-4">HR Letter Extractor</h2>
          <p className="text-gray-600">
            Extract information from HR documents using advanced OCR and AI
            processing
          </p>
        </Link>
        <Link
          to="/doc-chat"
          className="p-6 border rounded-lg shadow-lg hover:shadow-xl transition-shadow"
        >
          <h2 className="text-2xl font-semibold mb-4">Documentation Chat</h2>
          <p className="text-gray-600">
            Chat with your technical documentation using AI-powered analysis
          </p>
        </Link>
      </div>
    </div>
  );
}

export default App;
