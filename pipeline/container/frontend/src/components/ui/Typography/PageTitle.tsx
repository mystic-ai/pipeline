import React from "react";
import { PropsWithChildren } from "react";
import { cn } from "../../../utils/class-names";

interface Props extends PropsWithChildren {
  className?: string;
}
export function PageTitle({ children, className }: Props): JSX.Element {
  return (
    <h1
      className={cn(
        "font-bold text-gray-800 dark:text-gray-100 text-display_xs",
        className
      )}
    >
      {children}
    </h1>
  );
}
