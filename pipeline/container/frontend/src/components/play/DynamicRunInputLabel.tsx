import React from "react";
import { PropsWithChildren } from "react";
import { twMerge } from "../../utils/class-names";

type Props = {
  id: string;
  label: string;
  className?: string;
} & PropsWithChildren;

export function DynamicRunInputLabel({
  id,
  label,
  children,
  className,
}: Props): JSX.Element {
  return (
    <label htmlFor={id} className={twMerge("flex gap-1", className)}>
      <span className="text-sm font-medium text-gray-700 dark:text-gray-300 whitespace-nowrap">
        {label}
      </span>
      <span className="text-sm font-medium text-gray-500">:</span>
      <span className="text-sm font-medium text-gray-500 whitespace-nowrap">
        {children}
      </span>
    </label>
  );
}
