"use client";

import React, { useState, useEffect } from "react";
import FileUpload from "@/components/FileUpload";
import ChatInterface from "@/components/ChatInterface";
import { DocumentStatus } from "@/types/chat";

export default function DocumentationChat() {
  const [documents, setDocuments] = useState<DocumentStatus[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    console.log("Current documents state:", documents);
  }, [documents]);

  const handleFileSelect = async (file: File) => {
    console.log("File selected:", {
      name: file.name,
      type: file.type,
      size: `${(file.size / 1024 / 1024).toFixed(2)} MB`,
    });

    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      console.log("Uploading file to API...");
      const response = await fetch("/api/doc-upload", {
        method: "POST",
        body: formData,
      });

      console.log("Upload response status:", response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error("Upload error:", errorText);
        throw new Error(`Failed to upload document: ${errorText}`);
      }

      const data = await response.json();
      console.log("Upload successful:", data);

      setDocuments((prev) => {
        const newDocs = [...prev, data];
        console.log("Updated documents list:", newDocs);
        return newDocs;
      });
    } catch (err) {
      console.error("Error during upload:", err);
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">Documentation Chat</h1>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-1">
          <FileUpload
            onFileSelect={handleFileSelect}
            acceptedTypes={["PDF", "MD", "TXT"]}
          />
          {isUploading && (
            <div className="mt-4 p-4 bg-blue-50 rounded-md">
              <div className="flex items-center space-x-3">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-500"></div>
                <p className="text-blue-600">Uploading document...</p>
              </div>
            </div>
          )}
          {error && (
            <div className="mt-4 p-4 bg-red-50 text-red-600 rounded-md">
              {error}
            </div>
          )}
          {documents.length > 0 && (
            <div className="mt-4">
              <h3 className="text-lg font-semibold mb-2">Uploaded Documents</h3>
              <ul className="space-y-2">
                {documents.map((doc) => (
                  <li
                    key={doc.id}
                    className="p-2 bg-gray-50 rounded-md text-sm flex justify-between items-center"
                  >
                    <span className="truncate">{doc.filename}</span>
                    <span
                      className={`text-xs px-2 py-1 rounded-full ${
                        doc.status === "ready"
                          ? "bg-green-100 text-green-800"
                          : doc.status === "processing"
                          ? "bg-yellow-100 text-yellow-800"
                          : "bg-red-100 text-red-800"
                      }`}
                    >
                      {doc.status}
                    </span>
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
