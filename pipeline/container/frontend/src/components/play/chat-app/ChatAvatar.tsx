import React from "react";
import { UserProfileBox } from "../../ui/UserProfileBox";
import { GetPipelineResponse } from "../../../types";

interface Props {
  pipeline?: GetPipelineResponse;
}

export function ChatAvatar({ pipeline }: Props): JSX.Element {
  // We pass both username and pipeline, first to be available is shown
  const name = pipeline?.name || "Anonymous";
  const avatarColor = "#a931f4";

  return (
    <div className={"h-8 w-8 rounded-full"} aria-hidden="true">
      <UserProfileBox avatar_colour={avatarColor} username={name} />
    </div>
  );
}
