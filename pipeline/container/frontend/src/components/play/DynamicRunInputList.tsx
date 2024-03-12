import React from "react";
import { FieldErrors, FieldValues } from "react-hook-form";
import { DynamicRunInput } from "./DynamicRunInput";
import { DynamicRunInputLabel } from "./DynamicRunInputLabel";
import { cva, type VariantProps } from "class-variance-authority";
import { DynamicFieldData } from "../../types";
import { isObjectEmpty } from "../../utils/objects";
import { InputField } from "../ui/Inputs/InputField";

const groupStyles = cva("flex flex-col gap-2", {
  variants: {
    variant: {
      default: [""],
      minimal: ["[&_.label]:hidden", "[&_.bg-alternate]:bg-transparent"],
    },
    defaultVariants: {
      variant: "default",
    },
  },
});

export type DynamicRunInputListVariantProps = "default" | "minimal";

interface Props extends VariantProps<typeof groupStyles> {
  dynamicFields: DynamicFieldData[];
  isAuthed?: boolean;
  dirtyFields?: FieldValues;
  inputErrors?: FieldErrors<FieldValues>;
  variant?: DynamicRunInputListVariantProps;
}

export function DynamicRunInputList({
  dynamicFields,
  inputErrors,
  dirtyFields,
  isAuthed,
  variant,
}: Props): JSX.Element[] {
  return dynamicFields.map((d: DynamicFieldData, i: number) => {
    const isDict = d.subType === "dictionary" && d.dicts && d.dicts.length;
    return (
      <div key={d.label} className={groupStyles({ variant })}>
        <div className="flex flex-col gap-1 empty:hidden">
          <DynamicRunInput
            {...d}
            id={d.fieldName}
            // This needs to be wrapped in String() or else it will
            // loop and cause this to render the the dicts as an array with the dict
            // represented as json.
            defaultValue={isDict ? String(d.defaultValue) : d.defaultValue}
            errors={
              inputErrors && isObjectEmpty(inputErrors)
                ? undefined
                : inputErrors
            }
            isDirty={dirtyFields?.hasOwnProperty(d.fieldName)}
            optional={d.optional}
          />
        </div>

        {d.subType === "dictionary" && d.dicts && d.dicts.length ? (
          <InputField>
            <DynamicRunInputLabel label={d.label} id={d.id} className="label">
              dictionary
            </DynamicRunInputLabel>
            <div
              className={`flex flex-col gap-3 ${
                variant === "minimal" ? "" : "bg-alternate p-4 rounded-lg"
              }`}
            >
              <DynamicRunInputList
                dynamicFields={d.dicts}
                inputErrors={inputErrors}
              />
            </div>
          </InputField>
        ) : null}
      </div>
    );
  });
}
