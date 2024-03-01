import React from "react";
import {
  Tooltip as TooltipRoot,
  TooltipTrigger,
  TooltipContent,
  TooltipArrow,
} from "../../../contexts/TooltipProvider";
import {
  TooltipContentProps,
  type TooltipProps,
  Portal,
} from "@radix-ui/react-tooltip";
import { SyntheticEvent } from "react";

export interface Props extends TooltipProps {
  children: any;
  contentProps?: TooltipContentProps;
  content?: any;
  open?: boolean;
  defaultOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  onMouseDown?: (e: SyntheticEvent) => void;
  onMouseEnter?: (e: SyntheticEvent) => void;
  onMouseLeave?: (e: SyntheticEvent) => void;
  darkmode?: boolean;
}

export function Tooltip({
  children,
  contentProps,
  defaultOpen,
  onOpenChange,
  onMouseDown,
  onMouseEnter,
  onMouseLeave,
  content,
  darkmode,
  open,
  ...props
}: Props) {
  return (
    <TooltipRoot
      open={open}
      defaultOpen={defaultOpen}
      onOpenChange={onOpenChange}
      disableHoverableContent
      delayDuration={100}
      {...props}
    >
      <TooltipTrigger
        asChild
        onMouseDown={(e) => onMouseDown && onMouseDown(e)}
        onMouseEnter={(e) => onMouseEnter && onMouseEnter(e)}
        onMouseLeave={(e) => onMouseLeave && onMouseLeave(e)}
        onClick={(e) => onMouseDown && onMouseDown(e)}
      >
        {children}
      </TooltipTrigger>

      <Portal>
        {content ? (
          <TooltipContent
            alignOffset={-15}
            {...contentProps}
            className={`
              ${contentProps?.className ? contentProps?.className : ""}
              ${darkmode ? "bg-gray-600" : "bg-gray-800"}
            `}
          >
            {content}
            <TooltipArrow
              width={11}
              height={5}
              className={`tooltip-arrow ${
                darkmode ? "fill-gray-600" : "fill-gray-800"
              }`}
            />
          </TooltipContent>
        ) : null}
      </Portal>
    </TooltipRoot>
  );
}
