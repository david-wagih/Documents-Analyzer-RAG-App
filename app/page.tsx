import React from "react";
import Link from "next/link";

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-4rem)]">
      <h1 className="text-4xl font-bold text-center mb-8">
        Welcome to AI Tools Platform
      </h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <Link
          href="/hr-extractor"
          className="p-6 border rounded-lg shadow-lg hover:shadow-xl transition-shadow"
        >
          <h2 className="text-2xl font-semibold mb-4">HR Letter Extractor</h2>
          <p className="text-gray-600">
            Extract information from HR documents using advanced OCR and AI
            processing
          </p>
        </Link>
        <Link
          href="/doc-chat"
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
