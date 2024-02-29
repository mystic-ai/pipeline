import React from "react";
import { PropsWithChildren } from "react";
import { DescriptionText } from "../Typography/DescriptionText";
import { TitleText } from "../Typography/TitleText";

interface Props extends PropsWithChildren {
  size?: "sm" | "base";
}
export function EmptyResourceCard({
  size = "base",
  children,
}: Props): JSX.Element {
  return (
    <div className="border border-gray-300 dark:border-gray-900 rounded-lg overflow-auto flex flex-col justify-center items-center min-h-[12.5rem] gap-6 px-4 space-y-1 text-center">
      {size === "base" ? (
        <TitleText className="text-center">{children}</TitleText>
      ) : (
        <DescriptionText tag="p">{children}</DescriptionText>
      )}
    </div>
  );
}
