import React from "react";
import { useState, useRef, useEffect } from "react";
import { LayoutGroup } from "framer-motion";
import { Tab, TabSkeleton } from "./Tab";
import { v4 as uuidv4 } from "uuid";
import { cva, type VariantProps } from "class-variance-authority";
import { twMerge } from "../../../utils/class-names";

const tablist = cva("relative flex", {
  variants: {
    theme: {
      default: [
        "gap-3",
        "border-b",
        "border-gray-200",
        "dark:border-gray-700",
        "overflow-auto",
        "h-[2.8125rem]",
        "sm:h-auto",
        "sm:overflow-visible",
      ],
      code: ["gap-1"],
    },
    mainNav: {
      true: ["gap-3", "border-b-0"],
      false: [""],
      undefined: ["shadow-xs"],
    },
  },
});
// NOTES:
// https://codesandbox.io/s/animated-tabs-20k7l?file=/src/styles.css:715-747

type Tab = {
  title: string;
};

interface Props extends VariantProps<typeof tablist> {
  tabs: Tab[];
  theme?: "code" | "default";
  defaultActiveTabIndex?: number;
  handleTabClick?: (index: number) => void;
  className?: string;
  mainNav?: boolean;
  isLoading?: boolean;
  pathMatchIncludes?: boolean;
}

export function TabList({
  tabs,
  theme = "default",
  defaultActiveTabIndex,
  handleTabClick,
  className,
  mainNav,
  pathMatchIncludes = false,
  isLoading,
}: Props): JSX.Element {
  // Get current pathname

  // State
  const [[currentTab], setCurrentTab] = useState([0, 0]);

  // Get active tab
  let currentTabIndex: number = 0;

  // Wait for tabs props to be defined before setting Page state
  useEffect(() => {
    if (currentTabIndex === -1) currentTabIndex = -1;

    if (defaultActiveTabIndex !== undefined) {
      currentTabIndex = defaultActiveTabIndex;
    }

    setCurrentTab([currentTabIndex, currentTabIndex - currentTab]);
  }, [tabs]);

  // unique id for layoutID (framer)
  const ref = useRef({
    id: uuidv4(),
  });

  if (isLoading) {
    return (
      <ul
        className={twMerge(
          tablist({ theme, mainNav }),
          className,
          "!h-[2.8125rem]"
        )}
      >
        {tabs.map((tab, i) => {
          return (
            <TabSkeleton
              key={i}
              theme={theme}
              mainNav={mainNav}
              title={tab.title}
            />
          );
        })}
      </ul>
    );
  }

  return (
    <LayoutGroup>
      <ul className={twMerge(tablist({ theme, mainNav }), className)}>
        {tabs.map((tab, i) => {
          const isActive = i === currentTab;

          return (
            <Tab
              key={i}
              title={tab.title}
              isActive={isActive}
              theme={theme}
              id={ref.current.id}
              onClick={() => {
                setCurrentTab([i, i - currentTab]);
                handleTabClick && handleTabClick(i);
              }}
              mainNav={mainNav}
            />
          );
        })}
      </ul>
    </LayoutGroup>
  );
}
