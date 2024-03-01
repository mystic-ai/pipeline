import React from "react";
import { PropsWithChildren } from "react";

export function ButtonToggle({ children }: PropsWithChildren): JSX.Element {
  return (
    <div className="flex gap-px p-[2px] border border-gray-300 dark:border-gray-800 rounded-lg w-fit">
      {children}
    </div>
  );
}
