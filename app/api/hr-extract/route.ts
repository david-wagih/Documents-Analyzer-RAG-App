import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    console.log("HR Extract API called");
    const formData = await request.formData();

    const response = await fetch("http://localhost:8000/api/hr-extract", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      console.error("Backend error:", await response.text());
      throw new Error(`Failed to process document: ${response.statusText}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("HR Extract error:", error);
    return NextResponse.json(
      {
        error:
          error instanceof Error ? error.message : "Failed to process document",
      },
      { status: 500 }
    );
  }
}
