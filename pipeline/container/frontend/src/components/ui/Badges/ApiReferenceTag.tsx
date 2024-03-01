import React from "react";
import { PropsWithChildren } from "react";
import { Button } from "../Buttons/Button";

interface Props extends PropsWithChildren {
  inert?: boolean;
}
export function ApiReferenceTag({
  inert = true,
  children,
}: Props): JSX.Element {
  return (
    <Button
      size="tag"
      colorVariant="muted"
      tabIndex={inert ? -1 : undefined}
      inert={true}
      className="!bg-primary-100 dark:!bg-gray-700 px-1.5 font-medium rounded-[.1875rem] text-left select-text"
    >
      {children}
    </Button>
  );
}
