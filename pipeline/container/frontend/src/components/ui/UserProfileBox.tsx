import React from "react";
import { BlockSkeleton } from "./Skeletons/BlockSkeleton";

interface Props {
  username: string;
  avatar_colour: string;
}

export function UserProfileBox({
  username,
  avatar_colour,
}: Props): JSX.Element {
  // Constants
  const firstLetter = username.charAt(0).toUpperCase();

  return (
    <div
      className="
      flex rounded justify-center items-center
      w-8 h-8 text-white
      ring-0 [text-shadow:_0_0_2px_rgb(0_0_0_/_80%)]
      font-bold"
      style={{ backgroundColor: avatar_colour }}
    >
      {firstLetter}
    </div>
  );
}

export function UserProfileBoxSkeleton(): JSX.Element {
  return <BlockSkeleton height={27} width={27} />;
}
