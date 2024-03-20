import React from "react";

import { useEffect, useMemo, useState } from "react";
import { ChatAppForm } from "./ChatAppForm";
import { ChatAppMessages } from "./ChatAppMessages";
import { ChatAppSettingsDialog } from "./ChatAppSettingsDialog";
import { ChatPromptExamples } from "./ChatPromptExamples";
import {
  DynamicFieldData,
  GetPipelineResponse,
  PostRunPayload,
  RunInput,
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
import { postRun } from "../../../utils/queries/post-run";

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

  // State
  const [inputValue, setInputValue] = useState<string>("");
  const [showSettings, setShowSettings] = useState<boolean>(false);
  const [chatSettings, setChatSettings] = useState<{ [key: string]: any }>();
  const [chats, setChats] = useState<ChatMessages[]>([]);
  const [promptHistory, setPrompHistory] = useState<Prompt[]>();

  // We get a list of input variables that are of type dictionary,
  // We assume this is the settings for the chat
  const pipelineInputDictIOVariable = pipeline.input_variables.filter(
    (input) => input.run_io_type === "dictionary"
  );

  // This is state because we want to update this when the user changes the settings
  const initialDynamicFields = generateDynamicFieldsFromIOVariables(
    pipelineInputDictIOVariable
  );
  const [dynamicFields, setDynamicFields] =
    useState<DynamicFieldData[]>(initialDynamicFields);

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

    // Hide the settings dialog
    setShowSettings(false);
  }

  function handleChatSubmit(question: string) {
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
    setChats((chats) => [
      ...chats,
      { model: { value: "", createdAt: new Date(), isLoading: true } },
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

    postRun({ inputs })
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

          // Prepare to store in localstorage
          prompts.push({
            role: "assistant",
            content: run?.outputs[0].value[0].content,
          });

          // Push prompts in local storage
          localStorage.setItem(LS_KEY, JSON.stringify(prompts));
        }
      })
      .catch((error) => {
        console.log(error);
        notification.error({ title: "Error posting run." });
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

      // Set the chat settings state
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
