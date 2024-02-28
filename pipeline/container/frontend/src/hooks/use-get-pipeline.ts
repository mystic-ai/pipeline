import { UseQueryOptions, useQuery } from "@tanstack/react-query";
import { GetPipelineResponse } from "../types";
import { getPipeline } from "../lib/queries/get-pipeline";

export const queryKey = ["pipeline"];

type Options = {
  queryOptions?: UseQueryOptions<GetPipelineResponse>;
};
const useGetPipeline = ({ queryOptions }: Options = {}) => {
  return useQuery<GetPipelineResponse>({
    queryKey,
    queryFn: () => getPipeline(),
    ...(queryOptions ? queryOptions : {}),
  });
};

export default useGetPipeline;
