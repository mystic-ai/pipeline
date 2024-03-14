import React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import type { PropsWithChildren } from "react";
import { ChatAvatar } from "./ChatAvatar";
const timeFormat = new Intl.RelativeTimeFormat("en-US", { style: "narrow" });
import { TypingAnimation } from "./TypingAnimation";
import { ErrorAvatar } from "./ErrorAvatar";
import { GetPipelineResponse } from "../../../types";
import { DescriptionText } from "../../ui/Typography/DescriptionText";
import { ChatMarkdown } from "./ChatMarkdown";

const styles = cva(
  "p-4 flex flex-col gap-2 text-base font-medium rounded shadow-xs border border-gray-300 dark:border-gray-700 whitespace-pre-line",
  {
    variants: {
      variant: {
        received: ["bg-gray-100", "dark:bg-gray-900"],
        sent: ["bg-gray-100", "dark:bg-gray-900"],
        error: ["bg-red-100", "dark:bg-red-900", "border-red-300"],
      },
      defaultVariants: {
        variant: "default",
      },
    },
  }
);

interface Props extends PropsWithChildren, VariantProps<typeof styles> {
  createdAt: Date;
  isLoading?: boolean;
  responseTime?: number;
  ariaLabel?: string;
  pipeline?: GetPipelineResponse;
}

export function ChatBubble({
  children,
  variant,
  createdAt,
  isLoading,
  responseTime,
  ariaLabel,
  pipeline,
}: Props): JSX.Element {
  return (
    <div
      className="grid grid-cols-[theme(spacing.8),1fr] gap-3 mb-2"
      aria-label={ariaLabel}
    >
      {variant === "error" ? (
        <ErrorAvatar />
      ) : (
        <ChatAvatar pipeline={pipeline} />
      )}
      {/* avatar */}

      {/* content */}
      {isLoading ? (
        <div className="pt-4 pl-2">
          <TypingAnimation />
        </div>
      ) : (
        <div className={styles({ variant })} style={{ overflowAnchor: "none" }}>
          <ChatMarkdown content={String(children)}></ChatMarkdown>
          <div className="flex gap-3" aria-hidden="true">
            <DescriptionText className="text-xs  select-none">
              {createdAt.toLocaleTimeString()}
            </DescriptionText>

            {responseTime ? (
              <DescriptionText className="text-xs select-none">
                {responseTime > 1000
                  ? timeFormat.format(responseTime, "seconds").replace(",", ".")
                  : `${responseTime}ms`}
              </DescriptionText>
            ) : null}
          </div>
        </div>
      )}
    </div>
  );
}
