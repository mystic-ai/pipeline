import React from "react";
import { PropsWithChildren } from "react";
import { DescriptionText } from "../Typography/DescriptionText";

export function InputErrorText({ children }: PropsWithChildren): JSX.Element {
  return (
    <DescriptionText className="text-error-500 empty:hidden">
      {children}
    </DescriptionText>
  );
}
