import { UseQueryOptions, useQuery } from "@tanstack/react-query";
import { GetPipelineResponse, HTTPException } from "../types";
import { getPipeline } from "../utils/queries/get-pipeline";

export const queryKey = ["pipeline"];

type Options = {
  queryOptions?: UseQueryOptions<GetPipelineResponse, HTTPException>;
};
const useGetPipeline = ({ queryOptions }: Options = {}) => {
  return useQuery<GetPipelineResponse, HTTPException>({
    queryKey,
    queryFn: () => getPipeline(),
    ...(queryOptions ? queryOptions : {}),
  });
};

export default useGetPipeline;
