import { NextResponse } from "next/server";

export async function POST(request: Request) {
  console.log("Document upload API route called");

  try {
    const formData = await request.formData();
    const file = formData.get("file") as File;

    if (!file) {
      throw new Error("No file provided");
    }

    console.log("Received file:", {
      name: file.name,
      type: file.type,
      size: `${(file.size / 1024 / 1024).toFixed(2)} MB`,
    });

    const response = await fetch("http://localhost:8000/api/doc-upload", {
      method: "POST",
      body: formData,
    });

    console.log("Backend API response status:", response.status);

    if (!response.ok) {
      const errorData = await response.json();
      console.error("Backend API error:", errorData);
      throw new Error(errorData.detail || "Failed to upload document");
    }

    const data = await response.json();
    console.log("Backend API response:", data);

    return NextResponse.json(data);
  } catch (error) {
    console.error("Document upload API error:", error);
    return NextResponse.json(
      {
        error:
          error instanceof Error ? error.message : "Failed to upload document",
        details: error instanceof Error ? error.stack : undefined,
      },
      { status: 500 }
    );
  }
}
