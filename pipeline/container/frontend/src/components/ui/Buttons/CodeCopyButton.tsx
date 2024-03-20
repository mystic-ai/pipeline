import React from "react";

import { useEffect, useState } from "react";
import { Button, type ButtonSize } from "./Button";
import useCopyToClipboard from "../../../hooks/use-copy-to-clipboard";
import { Tooltip } from "../Tooltips/Tooltip";
import { twMerge } from "../../../utils/class-names";
import { IconCheckmark } from "../Icons/IconCheckmark";
import { IconCopy } from "../Icons/IconCopy";

interface Props {
  text: string;
  size: ButtonSize;
  className?: string;
}

export function CodeCopyButton({ text, size = "xxs", className = "" }: Props) {
  // Hooks
  const [_, copyToClipboard] = useCopyToClipboard();

  // State
  const [copied, setCopied] = useState<boolean>(false);
  const [tooltipOpen, setTooltipOpen] = useState<boolean>(false);

  // Effects
  useEffect(() => {
    const timer = setTimeout(() => {
      if (tooltipOpen) {
        setCopied(false);
        setTooltipOpen(false);
      }
    }, 1500);
    return () => clearTimeout(timer);
  }, [copied]);

  return (
    <Tooltip
      content={copied ? "Copied to clipboard!" : "Copy to clipboard"}
      contentProps={{
        align: "start",
      }}
      defaultOpen={false}
      onMouseDown={(e) => {
        e.stopPropagation();
        e.preventDefault();
        copyToClipboard(text);
        setCopied(true);
        setTooltipOpen(true);
      }}
      onMouseEnter={(e) => {
        e.stopPropagation();
        e.preventDefault();
        setTooltipOpen(true);
      }}
      onMouseLeave={(e) => {
        e.stopPropagation();
        e.preventDefault();
        setTooltipOpen(false);
      }}
      open={copied || tooltipOpen ? true : false}
    >
      <div>
        <Button
          size={size}
          colorVariant="custom"
          className={twMerge("ring-0 shadow-none group", className)}
          justIcon
        >
          <div className="pointer-events-none z-10">
            {copied ? (
              <IconCheckmark size={12} className="stroke-success-600" />
            ) : (
              <IconCopy
                size={12}
                className="stroke-gray-300 group-hover:stroke-white"
              />
            )}
          </div>
        </Button>
      </div>
    </Tooltip>
  );
}
