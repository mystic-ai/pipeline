export async function getFile(path: string): Promise<Blob> {
  try {
    const res = await fetch(`/v4/files/download/${path}`, {
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
    return await res.blob();
  } catch (error) {
    console.error("Error in getFile:", error);
    throw error;
  }
}
