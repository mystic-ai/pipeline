import { PropsWithChildren } from "react";

interface Props extends PropsWithChildren {}

export function InfoBlockList({ children }: Props): JSX.Element {
  return (
    <ul className="flex flex-col md:flex-row gap-4 md:gap-8 [&_h5]:whitespace-nowrap [&_p]:whitespace-nowrap [&>div]:w-fit overflow-auto">
      {children}
    </ul>
  );
}
