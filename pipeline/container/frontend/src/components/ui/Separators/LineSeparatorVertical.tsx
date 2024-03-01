import React from "react";
import { twMerge } from "../../../utils/class-names";

export function LineSeparatorVertical({ className }: { className?: string }) {
  return (
    <div
      className={twMerge(
        "flex-1 h-full w-px border-l border-gray-200 dark:border-gray-700",
        className
      )}
    />
  );
}
