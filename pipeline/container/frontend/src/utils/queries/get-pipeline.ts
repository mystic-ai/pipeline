import { GetPipelineResponse } from "../../types";

export async function getPipeline(): Promise<GetPipelineResponse> {
  try {
    const res = await fetch("/v4/container/pipeline", {
      method: "GET",
      headers: {
        Accept: "/*",
        "Content-Type": "application/json",
      },
    });
    if (!res.ok) {
      const errorResponse = await res.json();
      throw new Error(errorResponse.detail || "Unknown error occurred");
    }
    return await res.json();
  } catch (error) {
    console.error("Error in getPipeline:", error);

    throw error;
  }
}
