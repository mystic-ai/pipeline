import React from "react";
import { DescriptionText } from "../Typography/DescriptionText";
import { twMerge } from "../../../utils/class-names";

interface Props {
  id?: string;
  children: React.ReactNode;
  className?: string;
}

export function Label({ id, className, children }: Props): JSX.Element {
  return (
    <label htmlFor={id} className={twMerge("flex gap-1", className)}>
      <DescriptionText className="font-semibold">{children}</DescriptionText>
    </label>
  );
}
