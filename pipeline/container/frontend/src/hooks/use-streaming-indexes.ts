import { useMemo } from "react";
import { GetPipelineResponse, RunOutput } from "../types";

const useStreamingIndexes = (pipeline: GetPipelineResponse | undefined) => {
  // The indexes of stream type outputs in the outputs array
  if (!pipeline) return { isStreaming: false, streamOutputIndexes: [] };
  const streamOutputIndexes = useMemo(() => {
    return pipeline.output_variables.reduce((acc: number[], output, index) => {
      if (output.run_io_type === "stream") {
        acc.push(index);
      }
      return acc;
    }, []);
  }, [pipeline.output_variables]);

  const isStreaming = useMemo(() => {
    return streamOutputIndexes.length > 0;
  }, [streamOutputIndexes]);

  return { isStreaming, streamOutputIndexes };
};

export default useStreamingIndexes;
