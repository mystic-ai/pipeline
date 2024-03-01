import React from "react";
import { Button } from "../ui/Buttons/Button";

interface Props {
  url: string;
  alt: string;
}
export function PipelineRunImage(props: Props): JSX.Element {
  return (
    <div className="flex flex-col gap-4">
      <img src={props.url} alt={props.alt} className="w-full max-w-lg" />

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
