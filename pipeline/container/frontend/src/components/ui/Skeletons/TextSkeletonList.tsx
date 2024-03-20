import React from "react";
import { PropsWithChildren } from "react";
import { TextSkeleton } from "./TextSkeleton";

interface Props extends PropsWithChildren {
  loadingArrayLength: number;
}

const defaultLoadingLengths = [
  50, 70, 35, 90, 125, 50, 40, 100, 140, 160, 230, 60, 50, 70, 35, 90, 125, 50,
  40, 100, 140, 160, 230, 60, 50, 70, 35, 90, 125, 50, 40, 100, 140, 160, 230,
  60,
];

export const TextSkeletonList = ({ loadingArrayLength, children }: Props) => {
  return (
    <>
      {Array.from(Array(loadingArrayLength), (e, i) => (
        <TextSkeleton
          width={`${defaultLoadingLengths[i]}px`}
          key={i}
          className="block"
        >
          {children}
        </TextSkeleton>
      ))}
    </>
  );
};
