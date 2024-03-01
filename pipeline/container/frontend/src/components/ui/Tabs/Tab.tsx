import React from "react";
import { motion } from "framer-motion";
import { cva, type VariantProps } from "class-variance-authority";
import { TabText } from "./TabText";
import { TextSkeleton } from "../Skeletons/TextSkeleton";
import { Button, ButtonSize } from "../Buttons/Button";
import { twMerge } from "../../../utils/class-names";

const tab = cva("tab text-center", {
  variants: {
    theme: {
      default: ["gap-8", "h-[2.625rem]", "pb-4"],
      code: ["gap-0", "h-9", "pt-1.5", "pb-1", "px-1", "flex", "items-start"],
    },
    isActive: {
      true: [""],
      false: [""],
    },
    mainNav: {
      true: ["pb-3", "[&_a]:relative", "[&_a]:top-[-.375rem]"],
      false: [""],
    },
  },
  defaultVariants: {
    theme: "default",
  },
});

interface Props extends VariantProps<typeof tab> {
  id: string; // used to keep in sync with parent framer instance
  title: string;
  isActive?: boolean;
  theme?: "default" | "code";
  onClick?: () => void;
  mainNav?: boolean;
}

export function Tab({
  id,
  title,
  theme = "default",
  isActive = false,
  onClick,
  mainNav,
}: Props): JSX.Element {
  const buttonSize: ButtonSize =
    theme === "code" ? "custom" : mainNav ? "md" : "sm";

  return (
    <div
      className={twMerge(tab({ theme, isActive, mainNav }), "relative group")}
    >
      <Button
        colorVariant="muted"
        size={theme === "code" ? "custom" : "md"}
        className={`px-1 ${theme === "code" ? "hover:!bg-gray-900" : ""}`}
        onClick={onClick}
      >
        <TabText
          title={title}
          theme={theme}
          isActive={isActive}
          mainNav={mainNav}
        />
      </Button>
      {isActive && (
        <motion.div
          className={`absolute h-[.125rem] w-full bg-primary-button left-0 ${
            theme === "default" ? "-bottom-[.125rem]" : "bottom-0"
          }`}
          layoutId={id}
        />
      )}
    </div>
  );
}

type SkeletonProps = Omit<Props, "id" | "route" | "isActive" | "onClick">;

export function TabSkeleton({
  theme = "default",
  mainNav,
  title,
}: SkeletonProps): JSX.Element {
  const buttonSize: ButtonSize =
    theme === "code" ? "custom" : mainNav ? "md" : "sm";

  return (
    <Button colorVariant="muted" size={buttonSize} className="px-[.625rem]">
      <TextSkeleton>
        <TabText
          title={title}
          theme={theme}
          isActive={false}
          mainNav={mainNav}
        />
      </TextSkeleton>
    </Button>
  );
}
