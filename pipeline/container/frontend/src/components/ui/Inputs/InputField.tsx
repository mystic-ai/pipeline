import React from "react";
import { PropsWithChildren } from "react";

export function InputField({ children }: PropsWithChildren): JSX.Element {
  return <div className="flex flex-col gap-1.5">{children}</div>;
}
