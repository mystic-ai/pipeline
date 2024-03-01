import React from "react";
import { forwardRef } from "react";
import { IconCheckmark } from "../Icons/IconCheckmark";
import { Root, Indicator, type CheckboxProps } from "@radix-ui/react-checkbox";
import { twMerge } from "tailwind-merge";
import { Label } from "./Label";
import { DescriptionText } from "../Typography/DescriptionText";

interface Props extends CheckboxProps {
  id: string;
  labelTitle?: string;
  label?: string;
}

export const Checkbox = forwardRef(
  ({ id, labelTitle, label, ...props }: Props, forwardedRef) => {
    return (
      <label htmlFor={id} className="flex gap-2 items-start w-fit">
        <Root
          className={twMerge(
            `flex h-4 w-4 appearance-none items-center justify-center
            ring-1 ring-gray-300 rounded-[.25rem]
            bg-white
            hover:ring-2 hover:ring-gray-200 dark:hover:ring-primary-400
            focus:!ring-2 focus:ring-primary-200
            data-[state=checked]:!ring-primary-600
            !outline-none focus:!outline-none`
          )}
          id={id}
          {...props}
        >
          <Indicator className="mt-px">
            <IconCheckmark
              size={12}
              className="stroke-[3] stroke-primary-600"
            />
          </Indicator>
        </Root>

        {labelTitle || label ? (
          <Label id={id} className="flex flex-col select-none -mt-[.125rem]">
            {labelTitle}
            <DescriptionText className="block" tag="span">
              {label}
            </DescriptionText>
          </Label>
        ) : null}
      </label>
    );
  }
);
