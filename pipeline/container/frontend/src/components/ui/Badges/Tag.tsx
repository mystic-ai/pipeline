import { Button } from "@/components/ui/Buttons/Button";
import { PropsWithChildren } from "react";
import { twMerge } from "@/lib/helpers.className";

interface Props extends PropsWithChildren {
  inert?: boolean;
}
export function Tag({ inert = true, children }: Props): JSX.Element {
  return (
    <Button
      size="tag"
      tag="div"
      colorVariant="muted"
      tabIndex={inert ? -1 : undefined}
      inert={true}
      className={twMerge(
        "!bg-primary-100 dark:!bg-gray-700 font-medium rounded-[.1875rem] [&_svg]:stroke-gray-700 dark:[&_svg]:stroke-gray-100 ",
        !inert ? "select-text" : ""
      )}
    >
      {children}
    </Button>
  );
}
