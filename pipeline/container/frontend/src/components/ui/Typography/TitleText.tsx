import React from "react";
import { PropsWithChildren } from "react";
import { twMerge } from "../../../utils/class-names";

interface Props extends PropsWithChildren {
  className?: string;
  tag?: "h1" | "h2" | "h3" | "h4" | "h5" | "h6";
}

export function TitleText({
  children,
  tag,
  className,
  ...rest
}: Props): JSX.Element {
  const Component = tag ?? "h1";

  return (
    <Component
      className={twMerge(
        `text-xl font-bold
        gap-2 text-left
        text-gray-900 dark:text-gray-100
        `,
        className
      )}
      {...rest}
    >
      {children}
    </Component>
  );
}
