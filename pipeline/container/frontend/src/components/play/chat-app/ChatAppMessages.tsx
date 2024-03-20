import React from "react";
import { PropsWithChildren } from "react";
import type { ChatMessages } from "./ChatApp";
import { ChatBubble } from "./ChatBubble";
import { GetPipelineResponse } from "../../../types";

interface Props extends PropsWithChildren {
  chats: ChatMessages[];
  pipeline?: GetPipelineResponse;
}

export const ChatAppMessages = ({
  chats,
  pipeline,
  children,
}: Props): JSX.Element => {
  return (
    <div className="overflow-auto flex flex-col-reverse">
      <ul className="pt-4 pb-3 px-4" aria-live="polite">
        {chats && chats.length ? (
          chats.map((chat, index) => {
            return (
              <li key={index} className="flex flex-col">
                {chat.user && (
                  <ChatBubble
                    variant="sent"
                    {...chat.user}
                    ariaLabel={`you asked: ${chat.user.value}`}
                  >
                    {chat.user.value}
                  </ChatBubble>
                )}

                {chat.model && (
                  <ChatBubble
                    variant="received"
                    {...chat.model}
                    ariaLabel={`assistant answered: ${chat.model.value}`}
                    pipeline={pipeline}
                  >
                    {chat.model.value}
                  </ChatBubble>
                )}

                {chat.error && (
                  <ChatBubble
                    variant="error"
                    {...chat.error}
                    ariaLabel={`error: ${chat.error.value}`}
                    pipeline={pipeline}
                  >
                    {chat.error.value}
                  </ChatBubble>
                )}
              </li>
            );
          })
        ) : (
          <>{children}</>
        )}
      </ul>
    </div>
  );
};
