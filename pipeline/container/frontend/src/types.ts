import { RegisterOptions } from "react-hook-form";

// Error
export interface ErrorDetailResponse {
  // loc: [string, number];
  // msg: string;
  message: string;
  // type: string;
  parameter?: string;
}

export interface ErrorResponse {
  detail: ErrorDetailResponse | ErrorDetailResponse[];
}

// Pagination
export interface PaginatedPayload {
  skip: number;
  limit: number;
}

export interface PaginatedResponse<T> extends PaginatedPayload {
  total: number;
  data: T[];
}
// Sorting direction/order can be ascending/descending
export type BinarySortingOrder = "asc" | "desc";

export const defaultBinarySortingCycle: BinarySortingOrder[] = ["desc", "asc"];

export interface BinarySorting<SortingKey> {
  attribute: SortingKey;
  order: BinarySortingOrder;
}
// Sorting direction/order can be ascending/descending/no-sorting
export type TernarySortingOrder = "asc" | "desc" | null;

export const defaultTernarySortingCycle: TernarySortingOrder[] = [
  "desc",
  "asc",
  null,
];
export interface TernarySorting<SortingKey> {
  attribute: SortingKey;
  order: TernarySortingOrder;
}

export type Accelerator =
  | "nvidia_t4"
  | "nvidia_a100"
  | "nvidia_a100_80gb"
  | "nvidia_v100"
  | "nvidia_v100_32gb"
  | "nvidia_3090"
  | "nvidia_a16"
  | "nvidia_h100"
  | "nvidia_l4"
  | "nvidia_all"
  | "nvidia_a10"
  | "cpu";

export interface GetPipelineResponse {
  name: string;
  image: string;
  input_variables: IOVariable[];
  output_variables: IOVariable[];
  extras?: Record<string, any>;
}

export type RunState =
  | "created"
  | "routing"
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "no_resources_available"
  | "unknown";

export interface RunResult {
  outputs?: RunOutput[];
  inputs?: RunInput[];
  error?: RunError;
}

export interface GetRunResponse extends RunResult {
  id: string;
  created_at: number;
  updated_at: number;
  pipeline_id: string;
  state: RunState;
}

export interface RunOutputFile {
  name: string;
  path: string;
  url: string;
  size?: number;
}
export interface HTTPExceptionDetailObject {
  message?: string;
  [key: string]: any; // Optionally, allow any other properties
}

export interface HTTPException {
  status_code: number;
  detail: string | HTTPExceptionDetailObject;
  headers?: Record<string, string>;
}

export interface RunError {
  type: string;
  message: string;
  traceback?: string;
}

export interface RunOutput {
  type: RunIOType;
  value?: any;
  file?: RunOutputFile | null;
}

export interface Timespan {
  start: number;
  end: number;
}

export type RunIOType =
  | "integer"
  | "string"
  // A floating point number
  | "fp"
  | "dictionary"
  | "boolean"
  | "none"
  | "array"
  | "stream"
  // A pickled object
  | "pkl"
  | "file";

export const fileRunIOTypes = ["pkl", "file"];

export interface IOVariable {
  run_io_type: RunIOType;

  title?: string;
  description?: string;
  examples?: any[];
  gt?: number;
  ge?: number;
  lt?: number;
  le?: number;
  multiple_of?: number;
  allow_inf_nan?: boolean;
  max_digits?: number;
  decimal_places?: number;
  min_length?: number;
  max_length?: number;
  choices?: any[];
  dict_schema?: IOVariable[];
  default?: any;
  optional?: boolean;
}

export type subType =
  | "str"
  | "int"
  | "int-min-max"
  | "float-min-max"
  | "float-min"
  | "float-max"
  | "bool"
  | "options"
  | "file";

export type ControlType = "text" | "select" | "number" | "checkbox" | "file";

export interface SelectOption {
  label: string;
  value: string;
}
export interface BaseDynamicFieldData {
  id: string;
  label: string;
  inputType: ControlType;
  subType: RunIOType;
  fieldName: string;
  fieldDescription?: string;
  config?: RegisterOptions;
}
export interface DynamicFieldData {
  id: string;
  label: string; // Label, while the same value as fieldName, is used for display purposes
  inputType: ControlType;
  subType: RunIOType;
  fieldName: string;
  fieldDescription?: string;
  defaultValue?: any;
  options?: SelectOption[];
  config?: RegisterOptions;
  min?: number;
  max?: number;
  step?: number;
  dicts?: DynamicFieldData[];
  decimalPlaces?: number;
  optional?: boolean;
}

export interface GetPipelineInputsResponse {
  data: DynamicFieldData[];
}

export interface RunInput {
  type: RunIOType;
  value: any;

  file_name?: string;
  file_path?: string;
  file_url?: string;
}

export interface PostRunPayload {
  inputs: RunInput[];
}

export interface PostFileResponse {
  // The path of the file on the container
  path: string;
}

// Should new chunks be appended (e.g in chat streaming) or
// should they replace previous chunks (e.g. in image diffusion)
export type StreamingMode = "append" | "replace";
