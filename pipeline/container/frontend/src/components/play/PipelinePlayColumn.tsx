import React from "react";
import { PropsWithChildren, ReactNode } from "react";
import { CardAccordian } from "../ui/Cards/CardAccordian";
import { twMerge } from "../../utils/class-names";

interface Props extends PropsWithChildren {
  title?: ReactNode;
  className?: string;
}

export function PipelinePlayColumn({
  title,
  className,
  children,
}: Props): JSX.Element {
  return (
    <CardAccordian
      className={twMerge(
        "flex flex-col gap-8 w-full sm:w-vcol1 flex-1",
        className
      )}
      title={title}
      tag="h2"
    >
      {children}
    </CardAccordian>
  );
}
