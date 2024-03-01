import React from "react";
import { PropsWithChildren, ReactNode } from "react";
import { Card, type CardProps } from "./Card";
import * as Accordion from "@radix-ui/react-accordion";
import { LineSeparator } from "../Separators/LineSeparator";
import { IconChevronDown } from "../Icons/IconChevronDown";
import { Button } from "../Buttons/Button";
import { TitleText } from "../Typography/TitleText";

interface Props extends CardProps, PropsWithChildren {
  title: ReactNode;
  interactive?: boolean;
  className?: string;
  tag?: "h1" | "h2" | "h3" | "h4" | "h5" | "h6";
  defaultOpen?: boolean;
  titleChildren?: ReactNode;
}
export function CardAccordian({
  title,
  children,
  variant,
  interactive = false,
  className,
  tag = "h1",
  defaultOpen = false,
  titleChildren,
}: Props): JSX.Element {
  let value = "item-1";

  return (
    <Card className={`p-0 ${className}`} variant={variant}>
      <Accordion.Root
        type="single"
        collapsible={true}
        value={!interactive ? value : undefined}
        defaultValue={defaultOpen ? value : undefined}
      >
        <Accordion.Item value={value} className="flex flex-col">
          <Accordion.Header asChild>
            <TitleText tag={tag} className="text-base">
              <Accordion.Trigger
                asChild
                className="
                  group flex flex-1 justify-between
                  p-4 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition
                  focus-visible:relative focus-visible:z-10 focus-visible:!ring-2 focus-visible:!ring-primary-500
                  text-lg font-medium
                  [&>svg]:data-[state=open]:rotate-180 [&>svg]:data-[state=closed]:block
                  select-auto
                "
              >
                <Button
                  colorVariant="muted"
                  className={`w-full !rounded-[3.5px] font-bold ${
                    !interactive ? "hover:!bg-transparent cursor-auto" : ""
                  }`}
                  size="custom"
                  tabIndex={!interactive ? -1 : undefined}
                  tag={!interactive ? "div" : "button"}
                >
                  {title}

                  {interactive ? (
                    <IconChevronDown className="transition" />
                  ) : null}

                  {titleChildren ? titleChildren : null}
                </Button>
              </Accordion.Trigger>
            </TitleText>
          </Accordion.Header>

          <Accordion.Content
            className={`data-[state=open]:animate-slideDown data-[state=closed]:animate-slideUp ${
              interactive ? "overflow-hidden" : ""
            }`}
          >
            <LineSeparator />
            <div className="p-4 flex flex-col gap-8 flex-1">{children}</div>
          </Accordion.Content>
        </Accordion.Item>
      </Accordion.Root>
    </Card>
  );
}
