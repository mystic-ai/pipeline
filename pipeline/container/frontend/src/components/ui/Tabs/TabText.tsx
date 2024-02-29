import React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { CurlLogo, JavascriptLogo, PythonLogo } from "../Logos";
import { twMerge } from "../../../utils/class-names";

const tabText = cva(
  "flex gap-2 whitespace-nowrap transition-colors duration-100",
  {
    variants: {
      theme: {
        default: ["text-base", "font-semibold"],
        code: [
          "text-xs",
          "font-normal",
          "pt-[.25rem]",
          "pb-[.25rem]",
          "!text-gray-100",
        ],
      },
      isActive: {
        true: [""],
        false: [""],
      },
      darkmode: {
        true: [""],
        false: [""],
      },
      mainNav: {
        true: ["text-base", "font-semibold"],
        false: [""],
      },
    },
    compoundVariants: [
      // Main nav
      {
        theme: "default",
        isActive: false,
        mainNav: true,
        className: "text-gray-700 dark:text-gray-100",
      },
      {
        theme: "default",
        isActive: true,
        mainNav: true,
        className: "text-primary-900 dark:text-white",
      },

      // Other tabs (clean)
      {
        theme: "default",
        isActive: false,
        mainNav: undefined,
        className:
          "text-gray-900 dark:text-gray-400 group-hover:text-gray-900 dark:group-hover:text-gray-200",
      },
      {
        theme: "default",
        isActive: true,
        mainNav: undefined,
        className: "text-primary-700 dark:text-white",
      },
    ],
    defaultVariants: {
      theme: "default",
    },
  }
);

interface Props extends VariantProps<typeof tabText> {
  title: string;
  isActive?: boolean;
  theme?: "code" | "default";
  mainNav?: boolean;
}

export function TabText({
  title,
  theme = "default",
  isActive = false,
  mainNav,
}: Props): JSX.Element {
  let titleContent = <>{title}</>;

  if (theme === "code") {
    if (title.toLowerCase().includes("python")) {
      titleContent = (
        <>
          <PythonLogo size={15} />
          {title}
        </>
      );
    }
    if (title.toLowerCase().includes("javascript")) {
      titleContent = (
        <>
          <JavascriptLogo size={15} />
          {title}
        </>
      );
    }
    if (title.toLowerCase().includes("shell")) {
      titleContent = (
        <>
          <CurlLogo size={15} />
          {title}
        </>
      );
    }
  }

  return (
    <p className={twMerge(tabText({ theme, isActive, mainNav }))}>
      {titleContent}
    </p>
  );
}
