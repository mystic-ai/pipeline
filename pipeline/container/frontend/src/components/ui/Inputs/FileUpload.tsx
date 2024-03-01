import { useDropzone, Accept, FileRejection } from "react-dropzone";
import { IconUpload } from "@/components/ui/Icons/IconUpload";
import {
  IconWrapper,
  IconWrapperVariants,
} from "@/components/ui/Icons/IconWrapper";
import { useCallback, useEffect, useState } from "react";
import { cva } from "class-variance-authority";
import { twMerge } from "@/lib/helpers.className";
import { File } from "./File";
import { AnimatePresence, motion } from "framer-motion";
import { Label } from "./Label";
import { formatBytes } from "@/lib/helpers.bytes";
import { HintText } from "./HintText";
import { InputErrorText } from "./InputErrorText";
import { DescriptionText } from "../Typography/DescriptionText";

const fileupload = cva("file-upload px-3 py-4", {
  variants: {
    isDragActive: {
      true: "",
      false: "",
    },
    isDragAccept: {
      true: "",
      false: "",
    },
    isDragReject: {
      true: "",
      false: "",
    },
  },
  compoundVariants: [
    {
      isDragActive: true,
      isDragAccept: true,
      className: "bg-success-25 ring-2 ring-primary-500",
    },
    {
      isDragActive: true,
      isDragReject: true,
      className: "bg-error-25 ring-2 ring-red-500",
    },
  ],
});

function fileSizeValidator(file: File, maxLength: number) {
  if (file.size > maxLength) {
    return {
      code: "file-too-large",
      message: `File is larger than ${formatBytes(maxLength)}`,
    };
  }

  return null;
}

interface Props {
  label?: string;
  hintText?: string;
  status?: "clean" | "valid" | "invalid";
  invalidText?: string;
  defaultFiles?: File[];
  accept?: Accept;
  fileSizeLimit?: number;
  disabled?: boolean;
}

export function FileUpload({
  status,
  label,
  hintText,
  invalidText,
  defaultFiles = [],
  accept,
  fileSizeLimit,
  disabled,
}: Props): JSX.Element {
  const [files, setFiles] = useState<File[]>(defaultFiles);
  const [rejectedFiles, setRejectedFiles] = useState<FileRejection[]>();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    //TODO: if duplicate, show error/ rename duplicate file by appending count++

    // Do something with the files
    setFiles((oldArray) => [...acceptedFiles, ...oldArray]);
  }, []);

  // Hooks
  const {
    getRootProps,
    getInputProps,
    isDragActive,
    isDragAccept,
    isDragReject,
    fileRejections,
  } = useDropzone({
    validator: fileSizeLimit
      ? (file) => fileSizeValidator(file, fileSizeLimit)
      : undefined,
    onDrop,
    accept,
  });

  // Variants
  let iconWrapperVariant: IconWrapperVariants = "default";
  if (status === "invalid" || (isDragActive && isDragReject)) {
    iconWrapperVariant = "error";
  } else if (status === "valid" || (isDragActive && isDragAccept)) {
    iconWrapperVariant = "success";
  }

  // Effects
  useEffect(() => {
    if (fileRejections) {
      setRejectedFiles(() => fileRejections);
    }
  }, [fileRejections]);
  return (
    <div className="flex flex-col gap-2">
      {/* label */}
      {label ? <Label>{label}</Label> : null}

      {/* dropzone */}
      <div
        {...getRootProps({
          role: "button",
          "aria-label": "drag and drop area",
        })}
      >
        <div
          className={twMerge(
            fileupload({
              isDragActive,
              isDragAccept,
              isDragReject,
            })
          )}
          data-disabled={disabled}
        >
          <input {...getInputProps()} disabled={disabled || false} />
          <div className="flex flex-col gap-6 items-center">
            <IconWrapper variant={iconWrapperVariant}>
              <IconUpload size={20} />
            </IconWrapper>

            <div className="flex flex-col gap-1">
              {isDragAccept && (
                <>
                  <DescriptionText className="block text-success-500">
                    Some files will be accepted
                  </DescriptionText>
                </>
              )}
              {isDragReject && (
                <DescriptionText className="block text-red-500">
                  Some files will not be accepted
                </DescriptionText>
              )}
              {!isDragActive && (
                <DescriptionText className="block">
                  <span className="font-semibold ">Click to upload</span> or
                  drag and drop
                </DescriptionText>
              )}
              {accept && (
                <p className="text-sm text-gray-500 font-normal text-center">
                  {Object.keys(accept).map((keyName, i) => {
                    return (
                      <span key={keyName}>
                        {accept[keyName].map((ac) => {
                          return (
                            <span className="mr-1" key={ac}>
                              {ac}
                            </span>
                          );
                        })}
                      </span>
                    );
                  })}
                </p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* accepted files */}
      {files.length > 0 ? (
        <ul className="list-none space-y-2">
          <AnimatePresence>
            {files.map((file) => {
              return (
                <motion.li
                  key={file.name}
                  initial={{
                    y: 0,
                    opacity: 0,
                  }}
                  animate={{
                    y: 0,
                    opacity: 1,
                    transition: { duration: 0.15 },
                  }}
                  exit={{
                    opacity: 0,
                    transition: { duration: 0.15 },
                  }}
                  layout
                >
                  <File
                    key={file.name}
                    file={file}
                    onDelete={(file) => {
                      setFiles((prev) =>
                        prev.filter((f) => f.name !== file.name)
                      );
                    }}
                  />
                </motion.li>
              );
            })}
          </AnimatePresence>
        </ul>
      ) : null}

      {/* rejected files */}
      {rejectedFiles && rejectedFiles.length > 0 ? (
        <ul className="list-none space-y-2">
          <AnimatePresence>
            {rejectedFiles.map(({ file, errors }) => {
              return (
                <motion.li
                  key={file.name}
                  initial={{
                    y: 0,
                    opacity: 0,
                  }}
                  animate={{
                    y: 0,
                    opacity: 1,
                    transition: { duration: 0.15 },
                  }}
                  exit={{
                    opacity: 0,
                    transition: { duration: 0.15 },
                  }}
                  layout
                >
                  <File
                    key={file.name}
                    file={file}
                    fileErrors={errors}
                    onDelete={(file) => {
                      setRejectedFiles(
                        (prev) =>
                          prev && prev.filter((f) => f.file.name !== file.name)
                      );
                    }}
                  />
                </motion.li>
              );
            })}
          </AnimatePresence>
        </ul>
      ) : null}

      {/* hint message */}
      {status === "clean" || status === "valid" || hintText ? (
        <HintText>{hintText}</HintText>
      ) : null}

      {/* error message */}
      {status === "invalid" ? (
        <InputErrorText>{invalidText}</InputErrorText>
      ) : null}
    </div>
  );
}
