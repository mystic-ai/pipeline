import React from "react";
import { UserProfileBox } from "../../ui/UserProfileBox";

export function ErrorAvatar(): JSX.Element {
  return (
    <div
      className={"h-8 w-8 rounded-full"}
      aria-hidden="true"
      title="Error response"
    >
      <UserProfileBox username={"E"} avatar_colour={"rgb(239 68 68)"} />
    </div>
  );
}
