import { NextRequest, NextResponse } from "next/server";

const SERVER_URL = process.env.HB_SERVER_URL;
const API_KEY    = process.env.HB_API_KEY;
const DB_NAME    = process.env.HB_DB_NAME;
const NAMESPACE  = "cuad_clauses";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { slot_queries, top_k = 10 } = body;

    const resp = await fetch(
      `${SERVER_URL}/compose/search_slots/${DB_NAME}/${NAMESPACE}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": API_KEY!,
        },
        body: JSON.stringify({ slot_queries, top_k }),
      }
    );

    if (!resp.ok) {
      return NextResponse.json(
        { error: `HyperBinder error: ${resp.status}` },
        { status: resp.status }
      );
    }

    const data = await resp.json();
    return NextResponse.json(data);
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}