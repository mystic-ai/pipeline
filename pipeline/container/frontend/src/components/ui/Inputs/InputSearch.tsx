import { Button } from "@/components/ui/Buttons/Button";
import { IconSearch } from "@/components/ui/Icons/IconSearch";
import { IconXCircle } from "@/components/ui/Icons/IconXCircle";
import { cva, type VariantProps } from "class-variance-authority";
import { HTMLProps, useRef } from "react";
import { twMerge } from "@/lib/helpers.className";

const searchWrapper = cva("input block", {
  variants: {
    variant: {
      pipeline: [],
      explore: ["input--large"],
    },
  },
  defaultVariants: {
    variant: "pipeline",
  },
});

const searchInput = cva("input w-[100%] !pl-10 text-sm flex-grow", {
  variants: {
    variant: {
      pipeline: [""],
      explore: [""],
    },
  },
  defaultVariants: {
    variant: "pipeline",
  },
});

interface Props
  extends VariantProps<typeof searchInput>,
    HTMLProps<HTMLInputElement> {
  defaultValue?: string;
  onClear?: () => void;
}

export function InputSearch(props: Props) {
  const ref = useRef<HTMLInputElement>(null);
  const { defaultValue, variant, onClear, ...rest } = props;

  return (
    <label
      className="flex relative w-full h-full"
      htmlFor={props.id || "search"}
    >
      <div className="absolute top-0 bottom-0 left-4 flex h-full justify-center items-center flex-shrink-0 z-1">
        <IconSearch
          size={12}
          className="stroke-gray-700 dark:stroke-gray-300"
        />
      </div>

      <input
        type="text"
        className={searchInput({ variant })}
        id={props.id || "search"}
        defaultValue={defaultValue}
        ref={ref}
        {...rest}
      />

      {ref?.current?.value !== "" ? (
        <div className="flex cursor-pointer absolute right-0 top-0 h-full">
          <Button
            colorVariant="muted"
            justIcon
            size="lg"
            title="clear search field"
            aria-label="clear search field"
            onClick={() => {
              ref!.current!.value = "";
              onClear && onClear();
            }}
          >
            <IconXCircle size={16} className="stroke-gray-400" />
          </Button>
        </div>
      ) : null}
    </label>
  );
}
