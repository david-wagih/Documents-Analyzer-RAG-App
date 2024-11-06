"use client";

import React, { useState } from "react";
import FileUpload from "@/components/FileUpload";
import ChatInterface from "@/components/ChatInterface";
import { DocumentStatus } from "@/types/chat";

export default function DocumentationChat() {
  const [documents, setDocuments] = useState<DocumentStatus[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = async (file: File) => {
    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/doc-upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to upload document");
      }

      const data = await response.json();
      setDocuments((prev) => [...prev, data]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold mb-8">Documentation Chat</h1>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-1">
          <FileUpload
            onFileSelect={handleFileSelect}
            acceptedTypes={["PDF", "MD", "TXT"]}
          />
          {documents.length > 0 && (
            <div className="mt-4">
              <h3 className="text-lg font-semibold mb-2">Uploaded Documents</h3>
              <ul className="space-y-2">
                {documents.map((doc) => (
                  <li
                    key={doc.id}
                    className="p-2 bg-gray-50 rounded-md text-sm"
                  >
                    {doc.filename}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
        <div className="lg:col-span-2">
          <ChatInterface documents={documents} />
        </div>
      </div>
    </div>
  );
}
