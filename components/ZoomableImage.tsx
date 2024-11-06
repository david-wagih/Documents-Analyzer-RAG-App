"use client";

import React, { useState } from "react";

interface ZoomableImageProps {
  svg: string;
  alt?: string;
}

export default function ZoomableImage({ svg, alt }: ZoomableImageProps) {
  const [isZoomed, setIsZoomed] = useState(false);

  const openInNewTab = () => {
    // Create a blob from the SVG
    const blob = new Blob([svg], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    window.open(url, "_blank");
  };

  return (
    <>
      <div className="relative group">
        <div
          className="cursor-zoom-in transition-transform duration-200 hover:scale-[1.02]"
          onClick={() => setIsZoomed(true)}
          dangerouslySetInnerHTML={{ __html: svg }}
        />

        {/* Overlay buttons */}
        <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex gap-2">
          <button
            onClick={(e) => {
              e.stopPropagation();
              openInNewTab();
            }}
            className="bg-white/90 hover:bg-white p-2 rounded-lg shadow-lg text-gray-700 hover:text-gray-900 transition-all"
            title="Open in new tab"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
              />
            </svg>
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              setIsZoomed(true);
            }}
            className="bg-white/90 hover:bg-white p-2 rounded-lg shadow-lg text-gray-700 hover:text-gray-900 transition-all"
            title="Zoom in"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v6m3-3H7"
              />
            </svg>
          </button>
        </div>
      </div>

      {/* Modal for zoomed view */}
      {isZoomed && (
        <div
          className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4 backdrop-blur-sm"
          onClick={() => setIsZoomed(false)}
        >
          <div
            className="bg-white rounded-xl p-6 max-w-[90vw] max-h-[90vh] overflow-auto shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-end mb-2 sticky top-0 bg-white pb-2">
              <button
                onClick={() => setIsZoomed(false)}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>
            <div
              className="transform scale-150 origin-top-left min-w-[800px]"
              dangerouslySetInnerHTML={{ __html: svg }}
            />
          </div>
        </div>
      )}
    </>
  );
}
