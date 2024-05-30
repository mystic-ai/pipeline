import React from "react";
import { ReactNode } from "react";
import { isArray } from "./arrays";
import {
  ControlType,
  DynamicFieldData,
  IOVariable,
  RunIOType,
  fileRunIOTypes,
} from "../types";
import { ApiReferenceTag } from "../components/ui/Badges/ApiReferenceTag";
import { postFile } from "./queries/post-file";

export function isNullOrUndefined(value: any): value is null | undefined {
  return value === null || value === undefined;
}

export function generateFormDefaultValues(inputs: DynamicFieldData[]) {
  const defaultValues: { [key: string]: any } = {};

  inputs.forEach((input) => {
    const { fieldName, defaultValue } = input;
    defaultValues[fieldName] = defaultValue;
  });

  return defaultValues;
}

interface HandlePostFileProps {
  file: File;
}

async function handlePostFile({ file }: HandlePostFileProps) {
  const formData = new FormData();
  const encodedFileName = encodeURIComponent(file.name);
  formData.append(
    "pfile",
    new File([file], encodedFileName, { type: file.type })
  );

  try {
    const fileData = await postFile({ formData });
    return fileData;
  } catch (error) {
    throw new Error("Error while uploading the file.");
  }
}

// Generate Dict for a IOVariable
export async function generateDictFromDynamicFields({
  dynamicFields,
  data,
}: {
  dynamicFields: DynamicFieldData[];
  data: Record<string, any>;
}) {
  const inputs = [];

  for (const inputConfig of dynamicFields) {
    // We need to upload the file to the container before making a run
    if (fileRunIOTypes.includes(inputConfig.subType)) {
      const file: File = data[inputConfig.fieldName as keyof typeof data];

      try {
        const fileData = await handlePostFile({ file });
        inputs.push({
          type: "file",
          value: null,
          file_path: fileData.path,
        });
      } catch (error) {
        throw new Error("Error while uploading the file.");
      }
    } else if (inputConfig.subType === "dictionary") {
      const dictValues: { [key: string]: any } = {};
      const dictInputConfigs = inputConfig?.dicts || [];
      for (const dictInputConfig of dictInputConfigs) {
        const key = dictInputConfig.label;
        let value: any = data[dictInputConfig.fieldName as keyof typeof data];

        if (fileRunIOTypes.includes(dictInputConfig.subType)) {
          const file: File =
            data[dictInputConfig.fieldName as keyof typeof data];
          if (dictInputConfig.optional && !file) continue;
          const fileData = await handlePostFile({ file });
          value = { type: "file", value: null, file_url: fileData.path };
        }
        if (
          dictInputConfig.subType === "integer" ||
          dictInputConfig.subType === "fp"
        ) {
          value = Number(value);
        }

        if (dictInputConfig.subType === "boolean") {
          value = value === true ? true : false;
        }
        dictValues[key] = value;
      }

      inputs.push({
        type: "dictionary",
        value: dictValues,
      });
    } else if (inputConfig.subType === "array") {
      let value = data[inputConfig.fieldName as keyof typeof data];

      // Should be valid JSON at this point
      inputs.push({
        type: inputConfig.subType,
        value: value,
      });
    } else {
      let value: any = data[inputConfig.fieldName as keyof typeof data];
      if (inputConfig.subType === "integer" || inputConfig.subType === "fp") {
        value = Number(value);
      }
      inputs.push({
        type: inputConfig.subType,
        value,
      });
    }
  }
  return inputs;
}

// Generate meaningful default values for IOVariables
export function generateDefaultValue(variable: IOVariable): any {
  if (variable.default !== "" && !isNullOrUndefined(variable.default)) {
    return variable.default;
  }

  switch (variable.run_io_type) {
    case "integer": {
      if (!isNullOrUndefined(variable.lt)) {
        return variable.lt - 1;
      } else if (!isNullOrUndefined(variable.le)) {
        return variable.le;
      } else if (!isNullOrUndefined(variable.ge)) {
        return variable.ge;
      } else if (!isNullOrUndefined(variable.gt)) {
        return variable.gt + 1;
      } else {
        return 1;
      }
    }
    case "string": {
      if (variable.choices && variable.choices.length) {
        return variable.choices[0];
      }
      return "";
    }
    case "fp": {
      if (!isNullOrUndefined(variable.lt)) {
        return variable.lt - 1;
      } else if (!isNullOrUndefined(variable.le)) {
        return variable.le;
      } else if (!isNullOrUndefined(variable.ge)) {
        return variable.ge;
      } else if (!isNullOrUndefined(variable.gt)) {
        return variable.gt + 1;
      } else {
        return "0.00";
      }
    }
    case "dictionary": {
      if (variable.dict_schema) {
        const dictDefaults: Record<string, any> = {};
        for (const dictVar of variable.dict_schema) {
          dictDefaults[dictVar.title || ""] = generateDefaultValue(dictVar);
        }
        return dictDefaults;
      }
      return {};
    }
    case "boolean": {
      return false;
    }
    case "none": {
      return null;
    }
    case "array": {
      return [];
    }
    case "pkl": {
      return null; // or some appropriate default value for pickled objects
    }
    case "file": {
      return ""; // or some appropriate default value for file paths
    }
    default:
      return null;
  }
}

interface ReferenceDetailItem {
  title: "default" | "examples" | "requirements";
  examples?: ReferenceDetailRequiredItem[];
  children?: ReferenceDetailItem[];
}
interface ReferenceDetailRequiredItem {
  title?: string;
  examples?: ReactNode[];
}

// Generate strings representing the constrains of a IO Variable
export function generateReferenceDetails(
  props: IOVariable
): ReferenceDetailItem[] {
  let referenceDetailItems: ReferenceDetailItem[] = [];

  // If arbitrary dict value
  // Else normal default value
  if (
    typeof props.default === "object" &&
    !Array.isArray(props.default) &&
    props.default !== null
  ) {
    for (const [key, value] of Object.entries(props.default)) {
      if (!isNullOrUndefined(props.default)) {
        referenceDetailItems.push({
          title: "default",
          examples: [
            {
              title: key as string,
              examples: [<ApiReferenceTag>{value as string}</ApiReferenceTag>],
            },
          ],
        });
      }
    }
  } else {
    if (!isNullOrUndefined(props.default)) {
      let examples = String(props.default);

      if (isArray(props.default)) {
        examples = JSON.stringify(props.default, null, 2);
      }

      referenceDetailItems.push({
        title: "default",
        examples: [
          {
            examples: [<ApiReferenceTag>{examples}</ApiReferenceTag>],
          },
        ],
      });
    }
  }

  // Requirements
  const emptyRequirements: ReferenceDetailRequiredItem[] = [];
  let showRequirements = false;

  if (!isNullOrUndefined(props.max_length)) {
    emptyRequirements.push({
      title: "maximum length of the string",
      examples: [<ApiReferenceTag>{props.max_length}</ApiReferenceTag>],
    });
    showRequirements = true;
  }
  if (!isNullOrUndefined(props.max_digits)) {
    emptyRequirements.push({
      title: "maximum number of digits",
      examples: [<ApiReferenceTag>{props.max_digits}</ApiReferenceTag>],
    });
    showRequirements = true;
  }
  if (!isNullOrUndefined(props.gt)) {
    emptyRequirements.push({
      title: "greater than (gt)",
      examples: [<ApiReferenceTag>{props.gt}</ApiReferenceTag>],
    });
    showRequirements = true;
  }
  if (!isNullOrUndefined(props.ge)) {
    emptyRequirements.push({
      title: "greater than or equal to (ge)",
      examples: [<ApiReferenceTag>{props.ge}</ApiReferenceTag>],
    });
    showRequirements = true;
  }
  if (!isNullOrUndefined(props.lt)) {
    emptyRequirements.push({
      title: "Less than (lt)",
      examples: [<ApiReferenceTag>{props.lt}</ApiReferenceTag>],
    });
    showRequirements = true;
  }
  if (!isNullOrUndefined(props.le)) {
    emptyRequirements.push({
      title: "less than or equal to (le)",
      examples: [<ApiReferenceTag>{props.le}</ApiReferenceTag>],
    });
    showRequirements = true;
  }
  if (!isNullOrUndefined(props.multiple_of)) {
    emptyRequirements.push({
      title: "multiple of",
      examples: [<ApiReferenceTag>{props.multiple_of}</ApiReferenceTag>],
    });
    showRequirements = true;
  }
  if (!isNullOrUndefined(props.allow_inf_nan)) {
    emptyRequirements.push({
      title: "allow inf, -inf or nan values",
      examples: [
        <ApiReferenceTag>
          {props.allow_inf_nan ? "True" : "False"}
        </ApiReferenceTag>,
      ],
    });
    showRequirements = true;
  }
  if (!isNullOrUndefined(props.max_digits)) {
    emptyRequirements.push({
      title: "maximum number of digits",
      examples: [<ApiReferenceTag>{props.max_digits}</ApiReferenceTag>],
    });
    showRequirements = true;
  }
  if (!isNullOrUndefined(props.decimal_places)) {
    emptyRequirements.push({
      title: "maximum number of decimal places allowed",
      examples: [<ApiReferenceTag>{props.decimal_places}</ApiReferenceTag>],
    });
    showRequirements = true;
  }
  if (!isNullOrUndefined(props.min_length)) {
    emptyRequirements.push({
      title: "minimum length of the string",
      examples: [<ApiReferenceTag>{props.min_length}</ApiReferenceTag>],
    });
    showRequirements = true;
  }

  if (!isNullOrUndefined(props.choices) && props.choices.length) {
    emptyRequirements.push({
      title: "only allowed choices",
      examples: props.choices.map((c) => (
        <ApiReferenceTag key={c}>{c}</ApiReferenceTag>
      )),
    });
    showRequirements = true;
  }

  if (showRequirements) {
    referenceDetailItems.push({
      title: "requirements",
      examples: emptyRequirements,
    });
  }

  return referenceDetailItems;
}

// Generate a mock curl request for a pipeline
export function generateCurlRequest({
  inputVariables,
  apiToken,
  pipelineId,
}: {
  inputVariables: IOVariable[];
  apiToken: string;
  pipelineId?: string;
}): string {
  const inputs = inputVariables.map((input) => {
    if (input.run_io_type === "dictionary") {
      const outputArray =
        input.dict_schema &&
        input.dict_schema.flatMap((dict) => ({
          [dict.title as keyof typeof input]: generateDefaultValue(dict),
        }));

      const mergedValue = outputArray?.reduce((mergedObj, obj) => {
        return { ...mergedObj, ...obj };
      }, {});

      return {
        type: "dictionary",
        value: mergedValue,
      };
    }
    return {
      type: input.run_io_type,
      value: generateDefaultValue(input),
    };
  });

  const curlRequest = [
    `curl -X POST FILL ME IN LATERruns' \\`,
    "--header 'Content-Type: application/json' \\",
    "--data '{",
    `\t"inputs": `,
    `\t\t${JSON.stringify(inputs, null, "\t").replace(/\n( *)/g, "\n\t\t$1")}`,
    `\t}`,
    `'`,
  ].join("\n");

  return curlRequest;
}
const generateDisplayValue = (ioVariable: IOVariable) => {
  if (ioVariable.run_io_type === "string") return `"REPLACE WITH YOUR STRING"`;
  if (ioVariable.run_io_type === "file") return `open("my_file.txt", "rb")`;
  return generateDefaultValue(ioVariable);
};
const formatInputVariableForPython = (inputVariable: IOVariable): string => {
  if (inputVariable.run_io_type === "dictionary") {
    const start = "dict(";
    const end = "\t)";
    const middle = (inputVariable.dict_schema || []).map((variable) => {
      let defaultValue = variable.default;

      if (typeof variable.default === "boolean") {
        defaultValue = variable.default ? "True" : "False";
      } else if (typeof variable.default === "string") {
        defaultValue = `"""${variable.default}"""`;
      }

      return `\t\t${variable.title} = ${defaultValue},`;
    });

    return [start, ...middle, end].join("\n");
  }
  if (inputVariable.run_io_type === "array") {
    return `${JSON.stringify(inputVariable.default, null, "\t").replace(
      /\n( *)/g,
      "\n\t$1"
    )}`;
  }

  return inputVariable.default;
};

export function generatePythonRequest({
  inputVariables,
  apiToken,
  pipelineId,
}: {
  inputVariables: IOVariable[];
  apiToken: string;
  pipelineId?: string;
}): string {
  const formattedIOVariables = inputVariables.map((ioVariable, idx) => {
    return {
      ...ioVariable,
      title: ioVariable.title || `input-${idx}`,
      default: ioVariable.default || generateDisplayValue(ioVariable),
    };
  });

  const pythonRequest = [
    `from pipeline.cloud.pipelines import run_pipeline`,
    ``,
    `result = run_pipeline(`,
    `\t#pipeline pointer or ID`,
    `\t"${pipelineId}",`,
    ...formattedIOVariables.map(
      (ioVariable) =>
        `\t#:${ioVariable.title}\n\t${formatInputVariableForPython(
          ioVariable
        )},`
    ),
    `)`,
    ``,
    `print(result.outputs_formatted())`,
  ].join("\n");
  return pythonRequest;
}

const isIONumber = (run_io_type: RunIOType) =>
  run_io_type === "integer" || run_io_type === "fp";

const getIOControlType = (IOVariable: IOVariable): ControlType => {
  if (fileRunIOTypes.includes(IOVariable.run_io_type)) return "file";
  if (IOVariable.run_io_type === "boolean") return "checkbox";
  if (isIONumber(IOVariable.run_io_type)) return "number";
  if (IOVariable.choices) return "select";
  return "text";
};

const getIOMin = (IOVariable: IOVariable): number | undefined => {
  if (!isIONumber(IOVariable.run_io_type)) return;
  if (!isNullOrUndefined(IOVariable.gt)) {
    if (IOVariable.run_io_type === "fp") {
      if (IOVariable.multiple_of) return IOVariable.gt + IOVariable.multiple_of;
      else return IOVariable.gt + 0.1;
    } else {
      return IOVariable.gt + 1;
    }
  }
  if (!isNullOrUndefined(IOVariable.ge)) return IOVariable.ge;
};

const getIOMax = (IOVariable: IOVariable): number | undefined => {
  if (!isIONumber(IOVariable.run_io_type)) return;
  if (!isNullOrUndefined(IOVariable.lt)) {
    if (IOVariable.run_io_type === "fp") {
      if (!isNullOrUndefined(IOVariable.multiple_of))
        return IOVariable.lt - IOVariable.multiple_of;
      else return IOVariable.lt - 0.1;
    } else {
      return IOVariable.lt - 1;
    }
  }
  if (!isNullOrUndefined(IOVariable.le)) return IOVariable.le;
};

const getIOStep = (IOVariable: IOVariable): number | undefined => {
  if (!isIONumber(IOVariable.run_io_type)) return;
  if (IOVariable.multiple_of) return IOVariable.multiple_of;
  if (IOVariable.run_io_type === "fp") {
    if (!isNullOrUndefined(IOVariable.decimal_places)) {
      return 1 / Math.pow(10, IOVariable.decimal_places);
    } else {
      return 0.01;
    }
  }

  return 1;
};

// Take in IOVariables and return a DynamicFieldData array
// Format for HTML inputs
export const generateDynamicFieldsFromIOVariables = (
  IOVariables: IOVariable[]
): DynamicFieldData[] =>
  IOVariables.map((IOVariable, idx) => {
    return {
      id: `pipeline-play-input-${idx}`,
      label: IOVariable.title || `input-${idx}`,
      fieldName: IOVariable.title || `input-${idx}`,
      fieldDescription: IOVariable.description || "",
      inputType: getIOControlType(IOVariable),
      subType: IOVariable.run_io_type,
      options: IOVariable.choices?.map((choice) => ({
        label: choice,
        value: choice,
      })),
      max: getIOMax(IOVariable),
      min: getIOMin(IOVariable),
      step: getIOStep(IOVariable),
      dicts: IOVariable.dict_schema
        ? generateDynamicFieldsFromIOVariables(IOVariable.dict_schema)
        : undefined,
      defaultValue: generateDefaultValue(IOVariable),
      decimalPlaces: IOVariable.decimal_places,
      optional: IOVariable.optional,
    };
  });
