import React from "react";
import { DescriptionText } from "../Typography/DescriptionText";

export function OrSeparator({
  orientation = "horizontal",
}: {
  orientation?: "horizontal" | "vertical";
}) {
  return (
    <div className={`flex justify-center items-center gap-2`}>
      {orientation === "horizontal" ? (
        <hr className="flex-1 border-t border-gray-200 dark:border-gray-700 mt-[.125rem]" />
      ) : null}
      <DescriptionText>or</DescriptionText>
      {orientation === "horizontal" ? (
        <hr className="flex-1 border-t border-gray-200 dark:border-gray-700 mt-[.125rem]" />
      ) : null}
    </div>
  );
}
