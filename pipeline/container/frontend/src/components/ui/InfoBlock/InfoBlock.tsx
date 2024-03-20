import { ReactNode, PropsWithChildren } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { Tooltip } from "@/components/ui/Tooltips/Tooltip";
import { TextSkeleton } from "../Skeletons/TextSkeleton";
import { DescriptionText } from "../Typography/DescriptionText";

const infoBlock = cva("flex flex-col", {
  variants: {
    size: {
      sm: "gap-0",
      base: "gap-1",
    },
  },
});

interface Props extends PropsWithChildren, VariantProps<typeof infoBlock> {
  title: string;
  children: ReactNode;
  tooltipContent?: string;
  isLoading?: boolean;
}

export function InfoBlock({
  title,
  children,
  tooltipContent,
  isLoading,
  size = "base",
}: Props): JSX.Element {
  if (isLoading) {
    return (
      <li className={infoBlock({ size })} key={title}>
        <div className="mb-[3px] flex">
          <TextSkeleton>
            <p className="text-xs">{title}</p>
          </TextSkeleton>
        </div>
        <TextSkeleton>
          <p className="text-base">{children}</p>
        </TextSkeleton>
      </li>
    );
  }
  return (
    <li className={infoBlock({ size })} key={title}>
      <DescriptionText variant="secondary">{title}</DescriptionText>

      {tooltipContent ? (
        <Tooltip
          content={tooltipContent ? tooltipContent : undefined}
          contentProps={{ align: "center" }}
        >
          <div className="font-semibold text-base text-gray-700 dark:text-gray-100">
            {children}
          </div>
        </Tooltip>
      ) : (
        <div className="font-semibold text-base text-gray-700 dark:text-gray-100">
          {children}
        </div>
      )}
    </li>
  );
}
