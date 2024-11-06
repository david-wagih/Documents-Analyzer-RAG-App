import { NextResponse } from "next/server";

export async function GET() {
  try {
    const response = await fetch("http://localhost:8000/health");
    if (!response.ok) {
      throw new Error("Backend is not responding");
    }
    return NextResponse.json({ status: "ok" });
  } catch (error) {
    return NextResponse.json(
      { error: "Backend service is not available" },
      { status: 503 }
    );
  }
}
