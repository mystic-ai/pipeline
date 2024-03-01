import React from "react";
import { IconLoading2 } from "../Icons/IconLoading2";

export function Loading({ size }: { size: number }): JSX.Element {
  return (
    <span className="inline-block button-loading-icon">
      <IconLoading2 size={size} className="animate-spin" />
    </span>
  );
}
