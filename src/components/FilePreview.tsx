import React, { useState, useEffect } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/esm/Page/AnnotationLayer.css";
import "react-pdf/dist/esm/Page/TextLayer.css";

// Set up the worker
pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;

interface FilePreviewProps {
  file: File | null;
  url: string;
}

export default function FilePreview({ file, url }: FilePreviewProps) {
  const [numPages, setNumPages] = useState<number>(0);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [pdfData, setPdfData] = useState<string | ArrayBuffer | null>(null);

  useEffect(() => {
    if (file && file.type === "application/pdf") {
      const reader = new FileReader();
      reader.onload = (e) => {
        setPdfData(e.target?.result || null);
      };
      reader.readAsArrayBuffer(file);
    }
  }, [file]);

  const isPDF = file?.type === "application/pdf";
  const isImage = file?.type.startsWith("image/");

  function onDocumentLoadSuccess({ numPages }: { numPages: number }) {
    setNumPages(numPages);
  }

  if (!file) return null;

  return (
    <div className="my-6 p-4 bg-white rounded-lg shadow-lg">
      <div className="mb-4 flex justify-between items-center">
        <h3 className="text-lg font-semibold text-gray-700">File Preview</h3>
        <div className="text-sm text-gray-500">
          {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
        </div>
      </div>

      <div className="flex justify-center">
        {isPDF ? (
          <div className="w-full max-w-3xl">
            <Document
              file={pdfData}
              onLoadSuccess={onDocumentLoadSuccess}
              className="flex flex-col items-center"
              error={
                <div className="text-red-500">
                  Error loading PDF. Please try again.
                </div>
              }
              loading={
                <div className="flex items-center justify-center p-4">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                </div>
              }
            >
              <Page
                pageNumber={currentPage}
                width={Math.min(window.innerWidth * 0.8, 800)}
                className="shadow-lg"
                renderTextLayer={false}
                renderAnnotationLayer={false}
              />
            </Document>
            {numPages > 1 && (
              <div className="mt-4 flex justify-center items-center space-x-4">
                <button
                  onClick={() =>
                    setCurrentPage((prev) => Math.max(prev - 1, 1))
                  }
                  disabled={currentPage <= 1}
                  className="px-3 py-1 bg-blue-500 text-white rounded disabled:bg-gray-300"
                >
                  Previous
                </button>
                <span className="text-gray-600">
                  Page {currentPage} of {numPages}
                </span>
                <button
                  onClick={() =>
                    setCurrentPage((prev) => Math.min(prev + 1, numPages))
                  }
                  disabled={currentPage >= numPages}
                  className="px-3 py-1 bg-blue-500 text-white rounded disabled:bg-gray-300"
                >
                  Next
                </button>
              </div>
            )}
          </div>
        ) : isImage ? (
          <div className="max-w-3xl">
            <img
              src={url}
              alt="Preview"
              className="rounded-lg shadow-lg max-h-[600px] object-contain"
            />
          </div>
        ) : (
          <div className="text-gray-500">
            Preview not available for this file type
          </div>
        )}
      </div>
    </div>
  );
}
