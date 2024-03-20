import React from "react";
import { useState } from "react";
import SyntaxHighlighter from "react-syntax-highlighter";
import { cva, type VariantProps } from "class-variance-authority";
import { twMerge } from "../../../utils/class-names";
import { TabList } from "../Tabs/TabList";
import { CodeCopyButton } from "../Buttons/CodeCopyButton";
import { syntaxHighlighterStyleNightOwl } from "./syntaxHighlighterStyleNightOwl";

const code = cva("code-block", {
  variants: {
    hasCopyButton: {
      true: [""], // pr-10
      false: [""],
    },
    isActive: {
      true: ["block"],
      false: ["hidden"],
    },
    hasHeader: {
      true: ["rounded-[0_0_.3125rem_.3125rem]"],
      false: ["rounded"],
    },
    linePrefix: {
      $: ["[&_span]:before:content-['$_']", "[&_span]:before:text-gray-500"],
      undefined: [""],
    },
  },
});

export interface CodeTabsProps {
  title: string;
  code: any;
}

interface CodeProps extends VariantProps<typeof code> {
  hasHeader?: boolean;
  hasTabs: boolean;
  hasHeaderTitle?: boolean;
  hasCopyButton?: boolean;
  headerTitle?: string;
  tabs?: CodeTabsProps[];
  defaultActiveTabIndex?: number;
  className?: string;
}

export function Code({
  hasHeader = false,
  hasTabs,
  hasHeaderTitle = false,
  hasCopyButton,
  headerTitle,
  tabs,
  defaultActiveTabIndex,
  className,
  linePrefix,
}: CodeProps): JSX.Element {
  const [activeTabIndex, setActiveTabIndex] = useState<number>(
    defaultActiveTabIndex ? defaultActiveTabIndex : 0
  );

  return (
    <div className={twMerge(`flex flex-col rounded`, className)}>
      {/* header */}
      {hasHeader ? (
        <div className={`flex gap-4 px-2 rounded-[5px_5px_0_0] bg-gray-800`}>
          {/* Header title */}
          {hasHeaderTitle ? (
            <div className="flex items-center h-9">
              <h6 className={`text-sm font-medium text-white`}>
                {headerTitle}
              </h6>
            </div>
          ) : null}

          {/* Tabs */}
          {hasTabs && tabs?.length ? (
            <>
              <TabList
                theme="code"
                tabs={tabs}
                defaultActiveTabIndex={activeTabIndex}
                handleTabClick={(clickedTabIndex) =>
                  setActiveTabIndex(clickedTabIndex)
                }
              />
            </>
          ) : null}
        </div>
      ) : null}

      {/* Tabs */}
      {tabs?.map((tab, index) => {
        return (
          <div
            key={tab.code}
            className={code({
              hasCopyButton,
              isActive: activeTabIndex === index,
              hasHeader,
              linePrefix,
            })}
          >
            {hasCopyButton ? (
              <div className="absolute top-[.625rem] right-[.625rem]">
                <CodeCopyButton size="xxs" text={tab.code} />
              </div>
            ) : null}

            <SyntaxHighlighter
              language={
                tab.title.toLowerCase() === "shell"
                  ? "bash"
                  : tab.title.toLowerCase()
              }
              //@ts-ignore
              style={syntaxHighlighterStyleNightOwl}
              showLineNumbers={false}
            >
              {tab.code}
            </SyntaxHighlighter>
          </div>
        );
      })}
    </div>
  );
}
