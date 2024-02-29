import React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { forwardRef, ButtonHTMLAttributes } from "react";
import { twMerge } from "../../../utils/class-names";
import { IconLoading } from "../Icons/IconLoading";

const button = cva(
  "btn font-medium shadow-xs relative rounded leading-4 flex gap-2 items-center ring-1 transition [&_svg]:flex-shrink-0",
  {
    variants: {
      size: {
        custom: [""],
        tag: ["px-[.375rem]", "py-0", "gap-1", "text-xs", "leading-[1.3]"],
        xxs: [
          "h-4.5",
          "px-2",
          "py-1",
          "gap-1",
          "text-xs",
          "[&_.button-loading-icon]:w-3",
          "[&_.button-loading-icon]:h-3",
          "[&>svg]:w-3",
          "[&>svg]:h-3",
        ],
        xs: [
          "h-7",
          "px-2",
          "py-1",
          "gap-2",
          "text-sm",
          "[&>svg]:w-3",
          "[&>svg]:h-3",
        ],
        sm: [
          "h-9",
          "px-3",
          "py-2",
          "gap-2",
          "text-sm",
          "[&>svg]:w-4",
          "[&>svg]:h-4",
        ],
        md: [
          "h-10",
          "px-3",
          "py-2.5",
          "gap-2",
          "text-sm",
          "[&>svg]:w-5",
          "[&>svg]:h-5",
        ],
        lg: [
          "h-11",
          "px-4",
          "py-2.5",
          "gap-2",
          "text-base",
          "[&>svg]:w-5",
          "[&>svg]:h-5",
        ],
        xl: [
          "h-12",
          "leading-6",
          "py-3",
          "px-5",
          "gap-2",
          "text-base",
          "[&>svg]:w-5",
          "[&>svg]:h-5",
        ],
      },
      colorVariant: {
        primary: [
          // Background
          "bg-black",
          "hover:bg-gray-700",
          "dark:bg-white",
          "dark:hover:bg-gray-300",
          "disabled:hover:bg-black",
          "disabled:dark:hover:bg-white",
          // Text
          "text-white",
          "dark:text-black",
          // Border
          "ring-1",
          "ring-transparent",
          // Children
          "[&_svg]:!stroke-white",
          "dark:[&_svg]:!stroke-black",
        ],
        secondary: [
          // Background
          "bg-white",
          "dark:bg-gray-900",
          "disabled:hover:bg-transparent",
          "dark:hover:bg-gray-800",
          // Text
          "text-black",
          "dark:text-white",
          "hover:text-black",
          "hover:dark:text-white",
          // Border
          "ring-1",
          "ring-gray-300",
          "dark:ring-gray-700",
          "hover:ring-gray-400",
          "dark:hover:ring-gray-600",
          "disabled:hover:ring-gray-300",
          "disabled:dark:hover:ring-gray-400",
          // Children
          "[&_svg]:stroke-black",
          "dark:[&_svg]:stroke-white",
        ],
        muted: [
          // Other
          "shadow-none",
          // Background
          "bg-transparent",
          "hover:bg-gray-100",
          "dark:hover:bg-gray-900",
          "disabled:hover:!bg-transparent",
          // Text
          "text-gray-700",
          "dark:text-gray-100",
          "hover:text-gray-700",
          "hover:dark:text-gray-100",
          // Border
          "ring-0",
          "ring-transparent",
          // Children
          "[&_svg]:stroke-gray-700",
          "dark:[&_svg]:stroke-white",
        ],
        link: [
          // Other
          "shadow-none",
          // Background
          "bg-transparent",
          // Text
          "text-gray-700",
          "dark:text-gray-100",
          "hover:underline",
          "disabled:no-underline",
          "disabled:hover:!no-underline",
          // Border
          "ring-0",
          "ring-transparent",
          // Children
          "[&_svg]:stroke-gray-700",
          "dark:[&_svg]:stroke-white",
        ],
        "link-dark": [
          // Other
          "shadow-none",
          // Background
          "bg-transparent",
          // Text
          "text-gray-100",
          "hover:underline",
          "disabled:no-underline",
          "disabled:hover:!no-underline",
          // Border
          "ring-0",
          "ring-transparent",
          // Children
          "[&_svg]:stroke-white",
        ],
        "primary-dark": [
          // Background
          "bg-mystic-purple",
          "hover:!bg-mystic-purple-hover",
          "disabled:hover:brightness-100",
          // Text
          "text-white",
          // Border
          "ring-1",
          "ring-transparent",
          // Children
          "[&_svg]:stroke-white",
        ],
        "secondary-dark": [
          // Background
          "bg-transparent",
          "hover:!bg-white/10",
          "disabled:hover:bg-transparent",
          // Text
          "text-white",
          // Border
          "ring-1",
          "ring-gray-700",
          "hover:ring-white",
          "disabled:hover:ring-gray-300",
          // Children
          "[&_svg]:stroke-white",
        ],

        "primary-animated": [
          // Background
          "!bg-[linear-gradient(-45deg,#ee7752,#e73c7e,#23a6d5,#23d5ab)]",
          "hover:brightness-110",
          "disabled:hover:brightness-100",
          "animate-button",
          // Text
          "!text-white",
          "dark:!text-white",
          // Border
          "ring-0",
          // Children
          "[&_svg]:stroke-white",
        ],
        destructive: [
          // Background
          "bg-white",
          "dark:bg-black",
          "hover:bg-error-50",
          "dark:hover:bg-error-900/30",
          "disabled:hover:bg-white",
          "disabled:dark:hover:bg-black",
          // Text
          "text-error-700",
          "dark:text-error-400",
          // Border
          "ring-1",
          "ring-error-300",
          "dark:ring-error-400",
          // Children
          "[&_svg]:stroke-error-700",
          "dark:[&_svg]:stroke-error-400",
        ],
        custom: [""],
      },
      menuButton: {
        true: [
          "!shadow-none",
          "whitespace-nowrap",
          "text-gray-900",
          "!p-2",
          "!text-left",
          "w-full",
          "!justify-start",
          "!ring-transparent",
          "hover:bg-gray-100",
          "text-inherit",
        ],
        false: [""],
      },
      active: {
        true: [""],
        false: [""],
      },
      disabled: {
        true: ["cursor-not-allowed"],
        false: [""],
      },
      loading: {
        true: ["cursor-not-allowed"],
        false: [""],
      },
      justIcon: {
        true: [
          "!p-0",
          "relative",
          "overflow-hidden",
          "after:content-['']",
          "after:absolute",
          "after:inset-0",
        ],
        false: [""],
      },
      leftAlign: {
        true: ["justify-start"],
        false: ["justify-center"],
      },
      inert: {
        true: ["!shadow-none", "pointer-events-none"],
        false: [""],
      },
      darkmode: {
        true: [""],
        false: [""],
      },
    },
    compoundVariants: [
      {
        disabled: true,
        loading: false,
        className: ["opacity-50"],
      },
      {
        disabled: false,
        loading: true,
        className: [""],
      },
      {
        disabled: true,
        loading: true,
        className: ["cursor-not-allowed"],
      },

      // XSS
      {
        size: "xxs",
        justIcon: true,
        className: ["!w-4.5"],
      },
      // XS
      {
        size: "xs",
        justIcon: true,
        className: ["!w-7"],
      },
      // SM
      {
        size: "sm",
        justIcon: true,
        className: ["!w-9"],
      },
      // MD
      {
        size: "md",
        justIcon: true,
        className: ["!w-10"],
      },
      // LG
      {
        size: "lg",
        justIcon: true,
        className: ["!w-11"],
      },
      // XL
      {
        size: "xl",
        justIcon: true,
        className: ["!w-12"],
      },
      // Active
      {
        colorVariant: "primary",
        active: true,
        className: ["bg-gray-900", "dark:bg-gray-100"],
      },
      {
        colorVariant: "secondary",
        active: true,
        className: ["ring-gray-400", "dark:ring-gray-600"],
      },
      {
        colorVariant: "muted",
        active: true,
        className: ["bg-gray-100", "dark:!bg-gray-900"],
      },
      {
        colorVariant: "muted",
        darkmode: true,
        menuButton: false || undefined,
        className: ["bg-gray-800", "hover:!bg-gray-700", "text-white"],
      },
      // Mobile dropdown
      {
        colorVariant: "muted",
        darkmode: true,
        menuButton: true,
        className: ["bg-black", "hover:bg-gray-900", "text-white"],
      },

      {
        colorVariant: "secondary",
        justIcon: true,
        darkmode: false || undefined,
        className: ["after:bg-white"],
      },

      // Menu Buttons
      {
        darkmode: false,
        active: true,
        menuButton: true,
        className: ["bg-gray-100"],
      },
      {
        colorVariant: "link",
        darkmode: true,
        className: ["bg-black", "hover:!bg-black", "text-white"],
      },
      {
        colorVariant: "link",
        active: true,
        className: ["!underline"],
      },

      // {
      //   darkmode: true,
      //   menuButton: true,
      //   className: ["bg-black", "hover:bg-[#444444]", "text-white"],
      // },
      // {
      //   darkmode: true,
      //   menuButton: true,
      //   active: true,
      //   className: ["bg-[#444444]"],
      // },
    ],
  }
);

interface ButtonStyles extends VariantProps<typeof button> {}

export type ButtonHierachy =
  | "primary"
  | "secondary"
  | "muted"
  | "link"
  | "link-dark"
  | "primary-dark"
  | "secondary-dark"
  | "destructive"
  | "primary-animated"
  | "custom";

export type ButtonSize =
  | "custom"
  | "tag"
  | "xxs"
  | "xs"
  | "sm"
  | "md"
  | "lg"
  | "xl";

type TagProps = ButtonHTMLAttributes<HTMLButtonElement>;

// Generic type to generate HTML props based on its tag
export type ButtonProps = {
  size?: ButtonSize;
  colorVariant?: ButtonHierachy;
  children: JSX.Element | any;
  loading?: boolean;
  justIcon?: boolean;
  active?: boolean;
  disabled?: boolean;
  className?: string;
  menuButton?: boolean;
  inert?: boolean;
  leftAlign?: boolean;
  darkmode?: boolean;
  tag?: "button" | "div";
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps & TagProps>(
  (
    {
      size = "md",
      colorVariant = "primary",
      children,
      loading = false,
      justIcon = false,
      active = false,
      disabled = false,
      menuButton = false,
      leftAlign = false,
      inert = false,
      darkmode = false,
      tag,
      ...props
    }: ButtonProps & ButtonStyles,
    ref
  ): JSX.Element => {
    if (tag === "div") {
      return (
        <div
          {...props}
          ref={ref as React.Ref<HTMLDivElement>}
          className={twMerge(
            `btn-variant-is-${colorVariant}`,
            button({
              size,
              colorVariant,
              menuButton,
              active,
              disabled,
              justIcon,
              leftAlign,
              inert,
              darkmode,
              loading,
            }),
            props.className
          )}
        >
          <>
            {loading ? (
              <span className="button-loading-icon flex items-center ">
                <IconLoading size={20} className="animate-spin mx-auto" />
              </span>
            ) : null}

            {children}
          </>
        </div>
      );
    }

    return (
      <button
        {...props}
        ref={ref}
        disabled={disabled || loading ? true : false}
        className={twMerge(
          `btn-variant-is-${colorVariant}`,
          button({
            size,
            colorVariant,
            menuButton,
            active,
            disabled,
            justIcon,
            leftAlign,
            inert,
            darkmode,
            loading,
          }),
          props.className
        )}
      >
        <>
          {loading ? (
            <span className="button-loading-icon flex items-center ">
              <IconLoading size={20} className="animate-spin mx-auto" />
            </span>
          ) : null}

          {children}
        </>
      </button>
    );
  }
);
