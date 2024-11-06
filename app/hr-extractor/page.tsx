"use client";

import React, { useState } from "react";
import FileUpload from "@/components/FileUpload";
import FilePreview from "@/components/FilePreview";
import ProcessingStatus from "@/components/ProcessingStatus";
import ExtractedContent from "@/components/ExtractedContent";

export default function HRExtractor() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [extractedData, setExtractedData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileUrl, setFileUrl] = useState<string>("");

  const handleFileSelect = async (file: File) => {
    setSelectedFile(file);
    setFileUrl(URL.createObjectURL(file));
    setIsProcessing(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      console.log("Sending file to API:", file.name);
      const response = await fetch("/api/hr-extract", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("API Error:", errorText);
        throw new Error(`Failed to process document: ${response.statusText}`);
      }

      const data = await response.json();
      console.log("API Response:", data);
      setExtractedData(data);
    } catch (err) {
      console.error("Error:", err);
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsProcessing(false);
    }
  };

  // Cleanup URL when component unmounts
  React.useEffect(() => {
    return () => {
      if (fileUrl) {
        URL.revokeObjectURL(fileUrl);
      }
    };
  }, [fileUrl]);

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">HR Letter Extractor</h1>
      <FileUpload
        onFileSelect={handleFileSelect}
        acceptedTypes={["PDF", "PNG", "JPG", "JPEG"]}
      />
      {selectedFile && fileUrl && (
        <FilePreview file={selectedFile} url={fileUrl} />
      )}
      {isProcessing && <ProcessingStatus />}
      {error && (
        <div className="mt-4 p-4 bg-red-50 text-red-600 rounded-md">
          {error}
        </div>
      )}
      {extractedData && <ExtractedContent data={extractedData} />}
    </div>
  );
}
