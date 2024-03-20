import React from "react";
import { PropsWithChildren } from "react";

export function EmptyTableResourceCard({
  children,
}: PropsWithChildren): JSX.Element {
  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-auto flex flex-col justify-center items-center min-h-[12.5rem] gap-6 px-4">
      {children}
    </div>
  );
}
