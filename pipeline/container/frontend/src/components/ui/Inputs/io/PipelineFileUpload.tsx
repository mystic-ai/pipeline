import React from "react";

import { cva } from "class-variance-authority";
import { AnimatePresence, motion } from "framer-motion";
import { forwardRef, useCallback, useEffect, useState } from "react";
import { Accept, FileRejection, useDropzone } from "react-dropzone";
import { File } from "../File";
import { HintText } from "../HintText";
import { Label } from "../Label";
import { InputErrorText } from "../InputErrorText";
import { DescriptionText } from "../../Typography/DescriptionText";
import { formatBytes } from "../../../../utils/bytes";
import { BaseDynamicFieldData } from "../../../../types";
import { IconWrapper, IconWrapperVariants } from "../../Icons/IconWrapper";
import { twMerge } from "../../../../utils/class-names";
import { IconUpload } from "../../Icons/IconUpload";

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
    isFileStaged: {
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
    { isFileStaged: true, className: "hidden" },
  ],
});

function fileSizeValidator(file: File, maxLength: number, isAuthed = false) {
  if (file.size > maxLength) {
    let message = `File is larger than ${formatBytes(maxLength, 2, 1024)}.`;
    if (!isAuthed) {
      message += " Login to upload larger files.";
    }

    return {
      code: "file-too-large",
      message,
    };
  }

  return null;
}

interface Props extends Omit<BaseDynamicFieldData, "label"> {
  hintText?: string;
  status?: "clean" | "valid" | "invalid";
  invalidText?: string;
  defaultFiles?: File[];
  accept?: Accept;
  fileSizeLimit?: number;
  onChange: (file: File) => void;
  label?: string;
}

export const PipelineFileUpload = forwardRef<HTMLInputElement, Props>(
  (
    {
      status,
      label,
      hintText,
      invalidText,
      defaultFiles = [],
      accept,
      fileSizeLimit,
      onChange,
      id,
      fieldName,
      config,
      inputType,
      subType,
      fieldDescription,
      ...rest
    }: Props,
    ref
  ): JSX.Element => {
    const [files, setFiles] = useState<File[]>(defaultFiles);
    const [rejectedFiles, setRejectedFiles] = useState<FileRejection[]>();

    const onDrop = useCallback((acceptedFiles: File[]) => {
      // Since maxFiles is 1, acceptedFiles will only ever be
      // most recently dropped file
      setFiles(acceptedFiles);
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
      maxFiles: 1,
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

    useEffect(() => {
      onChange && onChange(files[0]);
    }, [files]);
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
                isFileStaged: files.length > 0,
              })
            )}
          >
            <input {...getInputProps()} {...rest} id={id} />
            <div className="flex flex-col gap-3 items-center">
              <IconWrapper variant={iconWrapperVariant}>
                <IconUpload size={20} />
              </IconWrapper>

              <div className="flex flex-col gap-1">
                {isDragAccept && (
                  <DescriptionText className="block text-success-500">
                    Some files will be accepted
                  </DescriptionText>
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
                            prev &&
                            prev.filter((f) => f.file.name !== file.name)
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
);
