import React from "react";
import { cva } from "class-variance-authority";
import { FileError } from "react-dropzone";
import { HintText } from "./HintText";
import { InputErrorText } from "./InputErrorText";
import { Label } from "./Label";
import { twMerge } from "../../../utils/class-names";
import { Button } from "../Buttons/Button";
import { IconTrash } from "../Icons/IconTrash";

const filestyle = cva(
  "file-item px-3 py-4 flex gap-3 justify-between items-center",
  {
    variants: {
      status: {
        valid: "input-valid",
        invalid: "input-invalid",
      },
    },
  }
);

interface Props {
  file: File;
  onDelete?: (file: File) => void;
  fileErrors?: FileError[];
}

export function File({ file, onDelete, fileErrors }: Props): JSX.Element {
  return (
    <div
      key={file.name}
      className={twMerge(
        filestyle({
          status: fileErrors && fileErrors.length ? "invalid" : "valid",
        })
      )}
    >
      <div className="flex flex-col">
        <Label>{file.name}</Label>
        {fileErrors && fileErrors.length ? (
          <InputErrorText>
            {fileErrors.map((text, i) => (
              <span key={i}>{text.message}</span>
            ))}
          </InputErrorText>
        ) : (
          <HintText>Staged for upload</HintText>
        )}
      </div>

      {onDelete ? (
        <Button
          colorVariant="secondary"
          justIcon
          size="xs"
          onClick={() => onDelete(file)}
        >
          <IconTrash size={20} />
        </Button>
      ) : null}
    </div>
  );
}
