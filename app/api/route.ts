import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const response = await fetch("http://localhost:8000/api/hr-extract", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Failed to process document");
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: "Failed to process document" },
      { status: 500 }
    );
  }
}
