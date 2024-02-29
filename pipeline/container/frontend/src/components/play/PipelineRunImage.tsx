import React from "react";
import { RunOutputFile } from "../../types";
import { Button } from "../ui/Buttons/Button";

export function PipelineRunImage(props: RunOutputFile): JSX.Element {
  return (
    <div className="flex flex-col gap-4">
      <img src={props.url} alt={props.name} className="w-full max-w-lg" />

      <a
        href={props.url}
        target="_blank"
        rel="noopener noreferrer"
        className="w-fit"
      >
        <Button colorVariant="primary" size="sm">
          View original
        </Button>
      </a>
    </div>
  );
}
