import React from "react";

import { useEffect, useMemo, useState } from "react";
import { ChatAppForm } from "./ChatAppForm";
import { ChatAppMessages } from "./ChatAppMessages";
import { ChatAppSettingsDialog } from "./ChatAppSettingsDialog";
import { ChatPromptExamples } from "./ChatPromptExamples";
import {
  DynamicFieldData,
  GetPipelineResponse,
  RunInput,
  RunResult,
} from "../../../types";
import {
  generateDynamicFieldsFromIOVariables,
  generateFormDefaultValues,
} from "../../../utils/ioVariables";
import { useNotification } from "../../ui/Notifications/Notifications";
import { Button } from "../../ui/Buttons/Button";
import { IconSend } from "../../ui/Icons/IconSend";
import { Tooltip } from "../../ui/Tooltips/Tooltip";
import { IconSettings4 } from "../../ui/Icons/IconSettings4";
import { IconTrash } from "../../ui/Icons/IconTrash";
import { DynamicFieldsForm } from "../DynamicFieldsForm";
import { postRun, streamPostRun } from "../../../utils/queries/post-run";
import useStreamingIndexes from "../../../hooks/use-streaming-indexes";

type ChatMessage = {
  value: string;
  createdAt: Date;
  isLoading?: boolean;
  responseTime?: number;
};
export type ChatMessages = {
  user?: ChatMessage;
  model?: ChatMessage;
  error?: ChatMessage;
};

interface Prompt {
  role: "user" | "system" | "assistant";
  content: string;
}

interface ChatAppProps {
  pipeline: GetPipelineResponse;
}

export default function ChatApp({ pipeline }: ChatAppProps): JSX.Element {
  // Constants
  const LS_KEY = `chat-app-prompt-history-${pipeline.image}`;

  // We get a list of input variables that are of type dictionary,
  // We assume this is the settings for the chat
  const pipelineInputDictIOVariable = pipeline.input_variables.filter(
    (input) => input.run_io_type === "dictionary"
  );

  // State
  const [inputValue, setInputValue] = useState<string>("");
  const [showSettings, setShowSettings] = useState<boolean>(false);
  const [chatSettings, setChatSettings] = useState<{ [key: string]: any }>();
  const [chats, setChats] = useState<ChatMessages[]>([]);
  const [promptHistory, setPrompHistory] = useState<Prompt[]>();

  // This is state because we want to update this when the user changes the settings
  const initialDynamicFields = generateDynamicFieldsFromIOVariables(
    pipelineInputDictIOVariable
  );
<<<<<<< Updated upstream
  const { isStreaming } = useStreamingIndexes(pipeline);
  // In rare occasions, the dynamic fields are not loaded yet
  let initialDictForSettings: DynamicFieldData[] | undefined = [];
  if (dynamicFields && dynamicFields.length > 0) {
    initialDictForSettings = dynamicFields.filter(
      (dynamicField) =>
        dynamicField.subType === "dictionary" &&
        dynamicField.dicts &&
        dynamicField.dicts.length > 0
    )[0].dicts;
  }
=======
  const [dynamicFields, setDynamicFields] =
    useState<DynamicFieldData[]>(initialDynamicFields);
>>>>>>> Stashed changes
  // Hooks
  const notification = useNotification();

  // Handlers
  function clearChat() {
    localStorage.removeItem(LS_KEY);
    setChats([]);
    setPrompHistory([]);
  }
  async function onSettingsSubmit(inputs: Array<any>) {
    notification.success({
      title: "Saved chat settings",
    });

    // Map the input value to the dynamic fields default value
    const updatedSettingsDynamicFields: DynamicFieldData[] | undefined =
      dynamicFields[0].dicts &&
      dynamicFields[0].dicts.map((dict, index) => {
        return {
          ...dict,
          defaultValue: inputs[index].value,
        };
      });

    // Update the dynamic fields state
    setDynamicFields((prev) => {
      prev[0].dicts = updatedSettingsDynamicFields;
      return prev;
    });

    setShowSettings(false);
  }
  function handleNewStreamChunk(chunk: RunResult) {
    const isChunkValueValid =
      chunk.outputs &&
      chunk.outputs[0].value &&
      chunk.outputs[0].value[0].content;

    if (!isChunkValueValid) {
      console.log("Chunk output is not in the expected format or missing.");
      return;
    }
    setChats((currentChats) => {
      if (currentChats.length === 0) {
        console.log("No chats to update.");
        return currentChats;
      }
      // Assuming `chats` is an array of chat messages and each chat message has a model property that contains the value.
      // First, make a shallow copy of the chats array to avoid directly mutating the state.
      const updatedChats = [...currentChats];

      // Check if there are any chats and if the last chat has a model property
      const isLastChatTypeModel =
        updatedChats.length > 0 && updatedChats[updatedChats.length - 1].model;
      if (!isLastChatTypeModel || !chunk.outputs) return updatedChats;
      // Get the last chat
      let lastChat = updatedChats[updatedChats.length - 1];

      // Check if lastChat.model is defined, otherwise use an empty string
      const lastChatValue = lastChat.model?.value || "";
      const newChunkContent = chunk.outputs[0].value[0].content;
      // Append the new chunk output to the last chat's model value (or use an empty string if model is undefined)
      const newChatValue = lastChatValue + newChunkContent;
      const now = new Date();
      const startTime = lastChat.model?.createdAt || now;

      // Update the last chat's model value with the new concatenated value
      lastChat = {
        ...lastChat,
        model: {
          ...lastChat.model,
          value: newChatValue,
          createdAt: now,
          isLoading: false,
          responseTime: now.getTime() - startTime.getTime(),
        },
      };

      // Update the last chat in the updatedChats array
      updatedChats[updatedChats.length - 1] = lastChat;

      // Update the state with the new chats array
      return updatedChats;
    });
  }

  async function handleChatSubmit(question: string) {
    if (!question) return;

    const userChat: ChatMessages = {
      user: { value: question, createdAt: new Date() },
    };
    let startTime = new Date();
    let prompts: Prompt[] = promptHistory || [];

    // Set user message in chat window optimistically
    setChats((chats) => [...chats, userChat]);

    // Prepare to store in localstorage
    const userPrompt: Prompt = {
      role: "user",
      content: question,
    };

    prompts.push(userPrompt);

    // Set a loading message in chat window optimistically
    // We will use createdAt to compute responseTime in new chunk handler
    setChats((chats) => [
      ...chats,
      { model: { value: "", createdAt: startTime, isLoading: true } },
    ]);

    // Build the request data
    const inputs: RunInput[] = [
      {
        type: "array",
        value: [prompts],
      },
      {
        type: "dictionary",
        value: chatSettings,
      },
    ];
    // We await so that the setChats call below gets the full content
    // of the lastChat after it has finished being streamed.
    if (isStreaming) {
      await streamPostRun({
        inputs,
        onNewChunk: handleNewStreamChunk,
      }).catch((error) => {
        notification.error({ title: "Error streaming run response." });
      });
    } else {
      await postRun({ inputs })
        .then((run) => {
          if (run && run.outputs) {
            setInputValue("");

            // Replace the AI loading message with the response AI message
            setChats((chats) => [
              ...chats.slice(0, -1),
              {
                model: {
                  //@ts-ignore
                  value: run?.outputs[0].value[0].content,
                  createdAt: new Date(),
                  responseTime: new Date().getTime() - startTime.getTime(),
                },
              },
            ]);
          }
        })
        .catch((error) => {
          console.log(error);
          notification.error({ title: "Error posting run." });
        });
    }
    // This just used to get the current chats and save stuff to localStorage
    setChats((chats) => {
      const lastChat = chats[chats.length - 1];
      if (!lastChat.model) return chats;
      prompts.push({
        role: "assistant",
        content: lastChat.model.value,
      });
      // Push prompts in local storage
      localStorage.setItem(LS_KEY, JSON.stringify(prompts));
      return chats;
    });
  }

  // On load set chat settings, update the chat settings
  useEffect(() => {
    // On load, and on close of settings dialog
    if (showSettings === false) {
      // Clear the chat settings state
      setChatSettings(undefined);

      // Build the key value pair for the chat settings
      const initialDictObject = generateFormDefaultValues(
        dynamicFields[0].dicts || []
      );
      setChatSettings(initialDictObject);
    }
  }, [showSettings]);

  useEffect(() => {
    const promptHistoryLS = localStorage.getItem(LS_KEY) || "[]";
    let promptHistory: Prompt[] = [];

    try {
      promptHistory = JSON.parse(promptHistoryLS);

      if (promptHistory) {
        // transform prompt[] into chat messages
        const chatsFromLS: ChatMessages[] = promptHistory.map((prompt) => {
          if (prompt.role === "user") {
            return {
              user: {
                value: prompt.content,
                createdAt: new Date(),
              },
            };
          } else if (prompt.role === "assistant") {
            return {
              model: {
                value: prompt.content,
                createdAt: new Date(),
              },
            };
          }
          return {};
        });

        setChats(chatsFromLS);
        setPrompHistory(promptHistory);
      }
    } catch (error) {
      console.log("error parsing prompts from local storage :>> ", error);
    }
  }, []);

  return (
    <div className="bg-white dark:bg-black border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
      <div className="grid grid-rows-[1fr,auto] h-[500px] col-span-2">
        <ChatAppMessages chats={chats} pipeline={pipeline}>
          {inputValue === "" ? (
            <div className="mx-auto max-w-[400px] pb-8">
              <ChatPromptExamples
                handleChoice={(question) => {
                  setInputValue(question);
                  handleChatSubmit(question);
                }}
              />
            </div>
          ) : null}
        </ChatAppMessages>

        <ChatAppForm
          handleSubmit={(question) => handleChatSubmit(question)}
          handleInputChange={(value) => setInputValue(value)}
        >
          <div className="flex flex-col gap-3">
            <Button
              aria-label="Send"
              type="submit"
              size="xl"
              colorVariant="primary"
              title="Send message"
              disabled={inputValue === ""}
            >
              Send <IconSend />
            </Button>

            <div className="flex gap-3">
              <Tooltip
                content={"Show chat settings"}
                contentProps={{ align: "start" }}
              >
                <div>
                  <Button
                    colorVariant="muted"
                    justIcon
                    size="xl"
                    onClick={() => setShowSettings(true)}
                    className="hover:bg-gray-200"
                  >
                    <IconSettings4 />
                  </Button>
                </div>
              </Tooltip>

              <Tooltip
                content={"Clear chat history"}
                contentProps={{ align: "start" }}
              >
                <div>
                  <Button
                    colorVariant="muted"
                    justIcon
                    size="xl"
                    onClick={() => clearChat()}
                    className="hover:bg-gray-200"
                  >
                    <IconTrash />
                  </Button>
                </div>
              </Tooltip>
            </div>
          </div>
        </ChatAppForm>
      </div>

      {/* {showSettings && pipelineInputDict ? ( */}
      {showSettings ? (
        <ChatAppSettingsDialog
          isOpen={showSettings}
          handleClose={() => setShowSettings(false)}
        >
          {dynamicFields &&
            dynamicFields[0] &&
            dynamicFields[0].dicts &&
            dynamicFields[0].dicts.length > 0 && (
              <DynamicFieldsForm
                onSubmitHandler={onSettingsSubmit}
                variant="minimal"
                dynamicFields={dynamicFields[0].dicts}
              >
                <div className="flex gap-3">
                  <Button colorVariant="primary" size="lg" type="submit">
                    Save settings
                  </Button>
                  <Button
                    colorVariant="secondary"
                    size="lg"
                    type="button"
                    onClick={() => setShowSettings(false)}
                  >
                    Close
                  </Button>
                </div>
              </DynamicFieldsForm>
            )}
        </ChatAppSettingsDialog>
      ) : null}
    </div>
  );
}
