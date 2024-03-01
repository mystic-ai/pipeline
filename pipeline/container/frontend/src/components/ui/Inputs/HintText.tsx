import React from "react";
import { PropsWithChildren } from "react";
import { DescriptionText } from "../Typography/DescriptionText";

export function HintText({ children }: PropsWithChildren): JSX.Element {
  return (
    <DescriptionText
      variant={"secondary"}
      className="min-w-[18.75rem] empty:hidden"
    >
      {children}
    </DescriptionText>
  );
}
