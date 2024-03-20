import React from "react";
import { Button } from "../ui/Buttons/Button";
import { Textarea } from "../ui/Inputs/Textarea";

interface Props {
  url?: string;
  alt: string;
  base64?: string;
}

export function PipelineRunImage(props: Props): JSX.Element {
  // Determine the source of the image based on the props provided
  const imageSrc = props.base64 || props.url || "some-dummy-stuff";
  const [imageError, setImageError] = React.useState(false);
  // If there's an error loading the image, render a Textarea instead
  if (imageError) {
    return (
      <Textarea
        id={imageSrc.substring(0, 10)}
        value={imageSrc}
        defaultValue={imageSrc}
        autoHeight
        readOnly
      />
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <img
        src={imageSrc}
        alt={props.alt}
        className="w-full max-w-lg"
        onError={() => setImageError(true)} // Set the error state if the image fails to load
      />

      {/* Render the link only if the URL is provided and no error occurred */}
      {props.url && !imageError && (
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
      )}
    </div>
  );
}
