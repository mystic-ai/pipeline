import React from "react";
import { Button } from "../../ui/Buttons/Button";
import { IconSend } from "../../ui/Icons/IconSend";

export const exampleChatPrompts = [
  "What is the average airspeed velocity of an unladen swallow?",
  "Can you explain the concept of quantum entanglement in simple terms?",
  "How do I convert a hexadecimal color code to a RGB color code?",
  "What is the difference between a Roth IRA and a traditional IRA?",
  "Can you give me a recipe for vegan chocolate chip cookies?",
  "What is the highest mountain peak in the solar system?",
  "How do I fix a leaky faucet in my bathroom sink?",
  "Can you explain the concept of confirmation bias and how it relates to decision making?",
  "What is the difference between a meteor and a meteorite?",
  "Can you give me a list of 5 fun facts about the human brain?",
];

interface Props {
  handleChoice: (choice: string) => void;
}

export function ChatPromptExamples({ handleChoice }: Props): JSX.Element {
  // return random 4 questions
  const randomQuestions = exampleChatPrompts
    .sort(() => Math.random() - Math.random())
    .slice(0, 2);

  return (
    <ul className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {randomQuestions.map((question) => (
        <Button
          key={question}
          colorVariant="secondary"
          className="flex items-start p-6 leading-relaxed justify-start text-left group"
          size="custom"
          leftAlign
          onClick={() => handleChoice(question)}
        >
          {question}
          <IconSend
            size={20}
            className="min-w-fit stroke-gray-200 group-hover:stroke-gray-600"
          />
        </Button>
      ))}
    </ul>
  );
}
